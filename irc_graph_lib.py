"""
irc_graph_lib.py — Postgres client for the IRC statutory graph.
Design: lazy connection, silent fallback, no startup-time DB access.
"""
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
_psycopg2 = None
_conn = None


def _import_psycopg2():
    global _psycopg2
    if _psycopg2 is None:
        try:
            import psycopg2 as _p
            _psycopg2 = _p
        except ImportError:
            logger.warning("psycopg2 not installed; graph enrichment disabled")
            _psycopg2 = False
    return _psycopg2 if _psycopg2 else None


def _get_conn():
    global _conn
    pg = _import_psycopg2()
    if not pg:
        return None
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    if _conn is not None:
        try:
            cur = _conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            return _conn
        except Exception:
            try: _conn.close()
            except: pass
            _conn = None
    try:
        _conn = pg.connect(db_url, connect_timeout=5)
        _conn.autocommit = True
        return _conn
    except Exception as e:
        logger.warning(f"Postgres connect failed: {e}")
        _conn = None
        return None


def is_available() -> bool:
    conn = _get_conn()
    if conn is None: return False
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM irc_sections LIMIT 1")
        count = cur.fetchone()[0]
        cur.close()
        return count > 0
    except Exception:
        return False


def get_section_count() -> int:
    conn = _get_conn()
    if conn is None: return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM irc_sections")
        n = cur.fetchone()[0]
        cur.close()
        return n
    except Exception: return 0


def get_cross_ref_count() -> int:
    conn = _get_conn()
    if conn is None: return 0
    try:
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM irc_cross_refs")
        n = cur.fetchone()[0]
        cur.close()
        return n
    except Exception: return 0


def get_section_context(canonical_id: str) -> str:
    if not canonical_id: return ""
    conn = _get_conn()
    if conn is None: return ""
    try:
        cur = conn.cursor()
        cur.execute("SELECT path::text, citation, short_title FROM irc_sections WHERE canonical_id=%s", (canonical_id,))
        row = cur.fetchone()
        if not row:
            cur.close(); return ""
        path, citation, short_title = row
        cur.execute("""
            SELECT level, citation, short_title FROM irc_sections
            WHERE path @> %s::ltree AND path <> %s::ltree
              AND level IN ('subpart','part','subchapter','chapter','subtitle')
            ORDER BY nlevel(path)
        """, (path, path))
        ancestors = cur.fetchall()
        cur.close()
        title_clip = (short_title or "")[:80]
        head = f"{citation}"
        if title_clip: head += f" ({title_clip})"
        if not ancestors: return head
        parts = []
        for level, cit, st in reversed(ancestors):
            st_clip = (st or "")[:60]
            label = f"{cit}"
            if st_clip and level in ('subpart','subchapter','part'):
                label += f" ({st_clip})"
            parts.append(label)
            if len(parts) >= 3: break
        return head + " -> " + " -> ".join(parts)
    except Exception as e:
        logger.warning(f"get_section_context failed for {canonical_id}: {e}")
        return ""


def get_outbound_refs(canonical_id, min_conf=1.0, in_corpus_only=True, limit=5) -> List[Dict]:
    if not canonical_id: return []
    conn = _get_conn()
    if conn is None: return []
    try:
        cur = conn.cursor()
        corpus_filter = "AND s_to.has_pinecone_chunks = true" if in_corpus_only else ""
        cur.execute(f"""
            SELECT DISTINCT ON (s_to.canonical_id)
                s_to.canonical_id, s_to.citation, s_to.short_title,
                cr.ref_kind, cr.confidence, s_to.has_pinecone_chunks
            FROM irc_cross_refs cr
            JOIN irc_sections s_from ON cr.from_path = s_from.path
            JOIN irc_sections s_to   ON cr.to_path   = s_to.path
            WHERE s_from.canonical_id = %s AND cr.confidence >= %s {corpus_filter}
            ORDER BY s_to.canonical_id, cr.confidence DESC
            LIMIT %s
        """, (canonical_id, min_conf, limit))
        results = [{
            "to_canonical_id": r[0], "citation": r[1],
            "short_title": r[2] or "", "ref_kind": r[3],
            "confidence": float(r[4]), "in_corpus": bool(r[5]),
        } for r in cur.fetchall()]
        cur.close()
        return results
    except Exception as e:
        logger.warning(f"get_outbound_refs failed for {canonical_id}: {e}")
        return []


def get_inbound_refs(canonical_id, min_conf=1.0, in_corpus_only=True, limit=5) -> List[Dict]:
    if not canonical_id: return []
    conn = _get_conn()
    if conn is None: return []
    try:
        cur = conn.cursor()
        corpus_filter = "AND s_from.has_pinecone_chunks = true" if in_corpus_only else ""
        cur.execute(f"""
            SELECT DISTINCT ON (s_from.canonical_id)
                s_from.canonical_id, s_from.citation, s_from.short_title,
                cr.ref_kind, cr.confidence, s_from.has_pinecone_chunks
            FROM irc_cross_refs cr
            JOIN irc_sections s_from ON cr.from_path = s_from.path
            JOIN irc_sections s_to   ON cr.to_path   = s_to.path
            WHERE s_to.canonical_id = %s AND cr.confidence >= %s {corpus_filter}
            ORDER BY s_from.canonical_id, cr.confidence DESC
            LIMIT %s
        """, (canonical_id, min_conf, limit))
        results = [{
            "from_canonical_id": r[0], "citation": r[1],
            "short_title": r[2] or "", "ref_kind": r[3],
            "confidence": float(r[4]), "in_corpus": bool(r[5]),
        } for r in cur.fetchall()]
        cur.close()
        return results
    except Exception as e:
        logger.warning(f"get_inbound_refs failed for {canonical_id}: {e}")
        return []


def resolve_chunk_to_section(chunk_metadata: dict) -> Optional[str]:
    if not isinstance(chunk_metadata, dict): return None
    for key in ("canonical_id", "doc_id", "source_doc", "source"):
        v = chunk_metadata.get(key)
        if v:
            s = str(v)
            for suffix in ("-current", "-pre1986", "-superseded", "-legacy"):
                if s.endswith(suffix):
                    s = s[:-len(suffix)]
                    break
            if s.startswith("IRC-"): return s
            if s.startswith("TREASREG-"):
                import re
                m = re.match(r'TREASREG-\d+\.(\d+)', s)
                if m: return f"IRC-{m.group(1)}"
            return None
    return None
