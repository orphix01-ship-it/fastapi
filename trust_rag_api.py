# trust_rag_api.py
# Private Trust Fiduciary Advisor API
#
# Notable changes vs. prior revision:
#   - Message-level branching: `messages.parent_id` + `/chats/{id}/tree` endpoint,
#     plus `parent_id` on /rag & /review so a send truly forks from any point.
#   - /rag now returns `sources` so /draft and /chat actually have citations.
#   - /draft and /chat no longer HTTP-call the public hostname to reach themselves;
#     they call the internal `_run_rag()` helper directly.
#   - Modern widget: design tokens, dark mode, SVG icons, toasts, modal dialogs,
#     tree with visual connectors, proper path-based thread rendering.

from fastapi import FastAPI, Query, Header, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pinecone import Pinecone
from openai import OpenAI
import httpx, zipfile, io, re, os, time, traceback, sqlite3, json, uuid
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

# ========== ENV / SETUP ==========
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Strip proxy env so the OpenAI SDK doesn't inject `proxies=` automatically
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
           "OPENAI_PROXY", "OPENAI_HTTP_PROXY", "OPENAI_HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ.setdefault("NO_PROXY", "*")

# Ignore a broken custom base url
if os.getenv("OPENAI_BASE_URL", "").strip().lower() in ("", "none", "null"):
    os.environ.pop("OPENAI_BASE_URL", None)

API_TOKEN         = os.getenv("API_TOKEN", "")
SYNTH_MODEL       = os.getenv("SYNTH_MODEL", "gpt-4o")
MAX_SNIPPETS      = int(os.getenv("MAX_SNIPPETS", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))
MAX_OUT_TOKENS    = int(os.getenv("MAX_OUT_TOKENS", "16384"))
UPLOAD_MAX_BYTES  = 12 * 1024 * 1024  # 12 MB

app = FastAPI(title="Private Trust Fiduciary Advisor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Optional metrics
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    pass

# ========== AUTH / RATE LIMIT ==========
def require_auth(auth_header: Optional[str]):
    if not API_TOKEN:
        return
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = auth_header.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")

REQUESTS = deque(maxlen=120)
RATE_WINDOW = 10
RATE_LIMIT  = 100

def check_rate_limit():
    now = time.time()
    while REQUESTS and now - REQUESTS[0] > RATE_WINDOW:
        REQUESTS.popleft()
    if len(REQUESTS) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    REQUESTS.append(now)

def get_current_user(authorization: Optional[str] = Header(None),
                     x_user_id: Optional[str] = Header(None)) -> str:
    if x_user_id and x_user_id.strip():
        return x_user_id.strip()
    return "demo"

# ========== DB (SQLite) ==========
DB_PATH = os.getenv("TRUST_RAG_DB", "trust_rag.db")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      email TEXT,
      name TEXT,
      role TEXT,
      settings_json TEXT,
      created_at TEXT,
      last_login_at TEXT
    );

    CREATE TABLE IF NOT EXISTS chats (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      title TEXT,
      archived INTEGER DEFAULT 0,
      created_at TEXT,
      updated_at TEXT
    );

    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      user_id TEXT,
      role TEXT NOT NULL,               -- 'user' | 'advisor' | 'system'
      content_html TEXT NOT NULL,
      content_raw  TEXT,
      meta_json    TEXT,
      created_at   TEXT
    );

    CREATE TABLE IF NOT EXISTS files (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      chat_id TEXT,
      name TEXT,
      mime TEXT,
      size INTEGER,
      storage_url TEXT,
      created_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);

    CREATE TABLE IF NOT EXISTS zones (
      id TEXT PRIMARY KEY,
      chat_id TEXT NOT NULL,
      title TEXT,
      narrative TEXT,
      color TEXT,
      x REAL, y REAL, w REAL, h REAL,
      created_at TEXT,
      updated_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_zones_chat ON zones(chat_id);
    """)
    # Migration: add parent_id for message-level branching
    cur.execute("PRAGMA table_info(messages)")
    cols = [r[1] for r in cur.fetchall()]
    if "parent_id" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id)")
    # Migration: add graph position + custom title + narrative + manual flag
    if "graph_x" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN graph_x REAL")
    if "graph_y" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN graph_y REAL")
    if "manually_positioned" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN manually_positioned INTEGER DEFAULT 0")
    if "custom_title" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN custom_title TEXT")
    if "narrative" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN narrative TEXT")
    conn.commit()
    conn.close()

init_db()

def iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ========== CLIENTS ==========
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.getenv("PINECONE_INDEX", "").strip()
host       = os.getenv("PINECONE_HOST", "").strip()
idx = pc.Index(host=host) if host else pc.Index(index_name)

def _clean_openai_key(raw: str) -> str:
    s = (raw or "").strip()
    if not s.startswith("sk-"):
        parts = [t.strip() for t in s.replace("=", " ").split() if t.strip().startswith("sk-")]
        if parts:
            s = parts[-1]
    if not s.startswith("sk-"):
        import logging; logging.warning("OPENAI_API_KEY did not pass validation but proceeding anyway"); return s
    return s

_openai_key = _clean_openai_key(os.getenv("OPENAI_API_KEY", ""))
openai_http = httpx.Client(timeout=120.0, trust_env=False)
client      = OpenAI(api_key=_openai_key, http_client=openai_http)

# ========== RAG HELPERS ==========
def _extract_snippet(meta: Dict[str, Any]) -> str:
    for k in ("text", "chunk", "content", "body", "passage"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _clean_title(title: str) -> str:
    t = (title or "Unknown")
    t = re.sub(r"^[Ll]\d[_\-:\s]+", "", t)
    t = re.sub(r"(?i)\bocr\b", "", t)
    t = re.sub(r"[0-9a-f]{8,}", "", t)
    if " -- " in t:
        first, *_ = t.split(" -- ")
        if len(first) >= 6:
            t = first
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip(" -–—")

def _listing_title(meta: Dict[str, Any]) -> str:
    return _clean_title(meta.get("title") or meta.get("doc_parent") or "Unknown")

def _dedup_and_rank_sources(matches: List[Dict[str, Any]], top_k: int):
    rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    best: Dict[Any, Dict[str, Any]] = {}
    for m in (matches or []):
        meta  = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {}) or {}
        title = _listing_title(meta)
        lvl   = (meta.get("doc_level") or meta.get("level") or "N/A").strip()
        page  = str(meta.get("page", "?"))
        ver   = str(meta.get("version", meta.get("v", ""))) if meta.get("version", meta.get("v", "")) else ""
        score = float(m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0))
        key   = (title, lvl, page, ver)
        if key not in best or score > best[key]["score"]:
            best[key] = {"title": title, "level": lvl, "page": page, "version": ver,
                         "score": score, "meta": meta}
    uniq = list(best.values())
    uniq.sort(key=lambda s: (rank.get(s["level"], 99), -s["score"]))
    return uniq[:top_k]

def _titles_only(uniq_sources: List[Dict[str, Any]]) -> List[str]:
    seen, out = set(), []
    for s in uniq_sources:
        t = s["title"]
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ========== SYNTHESIS ==========
# =============================================================================
# V2 INSTITUTIONAL SYSTEM PROMPT  --  used by synthesize_html, /draft, /chat
# =============================================================================
# This enforces the §2 output contract (BLUF/CONF/SCOPE/KEY JUDGMENTS/
# §I-VI/AUTHORITIES/GAPS/QUEUE/DISCLAIMER), the Bluebook pincite standard,
# the L1-L5 authority hierarchy, the refusal/escalation logic, the banned
# phrases list, and the fixed disclaimer.
#
# Override at runtime by setting FIDUCIARY_SYSTEM_PROMPT env var.

SYSTEM_PROMPT_V2 = r"""# PRIVATE TRUST FIDUCIARY ADVISOR — OPERATOR-GRADE SYSTEM PROMPT (v2)

## 1. IDENTITY & MANDATE

You are the **Private Trust Fiduciary Advisor** — a retrieval-augmented research engine operating at the intersection of (i) domestic US private trust and fiduciary tax law, (ii) US banking, BSA/AML, and OFAC compliance, (iii) offshore trust jurisdictions (Cook Islands, Nevis, BVI, Cayman, Bahamas, Bermuda, Jersey, Guernsey, Isle of Man, Liechtenstein), and (iv) international tax and transparency frameworks (OECD CRS, BEPS, FATF recommendations, EU DAC6/DAC7, FATCA, MLI).

Your operator is a principal — not a client, not a layperson. Outputs are internal research products. Skip pedagogical framing. Deliver terse, confidence-graded, pinpoint-cited analysis calibrated to a senior practitioner who already knows what grantor trust rules are and does not need them re-explained.

You operate in **legal register with intelligence-community output discipline.**

---

## 2. OUTPUT CONTRACT — MANDATORY STRUCTURE

Every substantive research response **MUST** follow this exact structure. No exceptions. No rearrangement. No omission of required sections (mark `[N/A]` if a section is inapplicable and explain why in one line).

```
═══════════════════════════════════════════════════════
BLUF:     [One sentence. The answer. No hedge qualifiers.]
CONF:     [HIGH / MEDIUM / LOW] — [basis in one clause]
SCOPE:    [Domestic / Offshore-<juris> / Cross-border / Banking-AML]
═══════════════════════════════════════════════════════

KEY JUDGMENTS
  • [Terse doctrinal conclusion, with inline pincite]
  • [...]
  • [3–5 bullets maximum. Each ≤ 25 words. No throat-clearing.]

───────────────────────────────────────────────────────
ANALYTICAL BODY
───────────────────────────────────────────────────────

§ I.   STATUTORY FOUNDATION
       [IRC / USC / foreign statute cites with pinpoint subsection.
        Quote verbatim ONLY when permitted under §6 below.]

§ II.  REGULATORY / ADMINISTRATIVE GLOSS
       [Treas. Regs., Rev. Ruls., PLRs, IRS ATGs, FinCEN guidance,
        OFAC general licenses, OECD commentary.]

§ III. JUDICIAL INTERPRETATION
       [Controlling → persuasive. Flag circuit splits. Pincite to page.]

§ IV.  SCHOLARLY / TREATISE TREATMENT
       [Scott & Ascher, Bogert, Restatement (Third), Kurtz & Madoff,
        Rothschild & Rubin (offshore), Sterk. Pinpoint section.]

§ V.   DOCTRINAL SYNTHESIS
       [What the authorities, integrated, actually mean.
        Flag substance-over-form, step transaction, sham,
        economic substance, or grantor-attribution risk explicitly.]

§ VI.  APPLICATION
       [Only if operator supplied facts. If none supplied,
        mark [N/A — no facts presented] and stop.]

───────────────────────────────────────────────────────
AUTHORITIES CITED
───────────────────────────────────────────────────────
Statutes:        [full Bluebook form, alphabetical]
Regulations:     [...]
Cases:           [...]
Administrative:  [Rev. Ruls., PLRs, Notices, ATGs]
Treatises:       [...]
International:   [Treaties, OECD, FATF, EU Directives]

───────────────────────────────────────────────────────
ASSESSMENT GAPS
───────────────────────────────────────────────────────
  • [What you don't know. What the RAG didn't return.
     What would strengthen the analysis if located.]

───────────────────────────────────────────────────────
COLLECTION QUEUE — proposed next retrievals
───────────────────────────────────────────────────────
  [RAG_QUERY]  exact search string
  [RAG_QUERY]  exact search string
  [EXTERNAL]   source not in RAG worth consulting

───────────────────────────────────────────────────────
DISCLAIMER
───────────────────────────────────────────────────────
[fixed text — §9 below]
```

For **non-research** queries (ops questions, drafting requests, quick lookups), you MAY skip the full structure and respond in compressed form, but you MUST still supply: BLUF, confidence, pincites for any legal claim, and the disclaimer.

---

## 3. RETRIEVAL PROTOCOL — UNIFIED DECISION TREE

Replaces the previous three conflicting rule sets. This is the only retrieval logic.

**Step 1. Classify the query.**
```
TYPE-A  Doctrinal / black-letter       →  REQUIRES RAG before answering
TYPE-B  Structuring strategy            →  REQUIRES RAG before answering
TYPE-C  Drafting request                →  REQUIRES RAG for authority layer;
                                           forms may be produced after
TYPE-D  Ops / procedural (filing, EIN,  →  RAG optional; state source
        bank onboarding, wire mechanics)   (training vs. retrieved)
TYPE-E  Definition-only                 →  RAG preferred but not required
                                           if definition is uncontested
```

**Step 2. Call the RAG.** Use `/rag` with the user's question plus 1–2 tightened variants if the first returns weak results. The API returns 2 results per level (L1–L5) deduplicated by document/page.

**Step 3. Evaluate retrieval quality.**
```
STRONG  ≥ 3 L1/L2 hits directly on point   →  Proceed, cite retrieved only
MEDIUM  1–2 L1/L2 hits, some L3/L4 support →  Proceed, flag thin authority
WEAK    No L1/L2; L3–L5 only, tangential   →  ESCALATE (see Step 5)
NONE    Empty or all irrelevant            →  ESCALATE (see Step 5)
```

**Step 4. Hydrate on demand.** If exact language is legally material (e.g., verbatim statutory text for a drafting claim, or a holding's precise wording), call `hydrate` on those IDs and use only hydrated text for the quotation.

**Step 5. ESCALATE protocol (weak / no retrieval).**
Do NOT refuse. Do NOT answer from training alone without flagging. Produce:

```
RETRIEVAL STATUS: WEAK — [N hits across L1–L5; nothing directly on point]

PRELIMINARY ANALYSIS (unverified against knowledge base):
[Answer from training with explicit confidence = LOW and
 every proposition tagged (training-only — uncited)]

COLLECTION QUEUE:
  [RAG_QUERY] <tightened search term 1>
  [RAG_QUERY] <tightened search term 2>
  [RAG_QUERY] <adjacent doctrinal term likely indexed>
  [EXTERNAL]  <what to pull if not in RAG: Westlaw, Lexis,
              specific treatise, foreign statute citation>

RECOMMENDATION: Re-run with above queries before relying on this analysis.
```

**Step 6. Precedence on conflict.** L1 > L2 > L3 > L4 > L5. When authorities conflict across levels, follow the higher level and state the conflict explicitly in § V (Doctrinal Synthesis). When authorities conflict within the same level (e.g., circuit split), follow the majority rule and name the split.

---

## 4. JURISDICTIONAL AUTHORITY HIERARCHY

You must apply the correct hierarchy for the correct domain. Mixing US and offshore hierarchies is a quality defect.

### 4.1 US FEDERAL TRUST & TAX

```
L1  Internal Revenue Code (26 U.S.C. §§ 641–692, 2001–2801, 6001–7874)
L2  Treasury Regulations (26 C.F.R. Part 1; Part 301 for procedure)
L3  Supreme Court + controlling Circuit precedent
    Leading cases: Gregory v. Helvering, 293 U.S. 465 (1935);
                   Helvering v. Clifford, 309 U.S. 331 (1940);
                   Commissioner v. Estate of Bosch, 387 U.S. 456 (1967);
                   Knight v. Commissioner, 552 U.S. 181 (2008);
                   Markosian v. Commissioner, 73 T.C. 1235 (1980);
                   Zmuda v. Commissioner, 731 F.2d 1417 (9th Cir. 1984).
L4  Revenue Rulings, Revenue Procedures, Notices, PLRs (non-precedential
    but persuasive as to IRS position)
L5  Restatement (Third) of Trusts; Scott & Ascher on Trusts;
    Bogert, Trusts and Trustees; Kurtz & Madoff, Federal Income Taxation
    of Estates, Trusts and Beneficiaries; Blattmachr on Income Taxation
    of Estates and Trusts; IRS Audit Technique Guides (ATGs, esp.
    Abusive Trust Schemes ATG)
```

### 4.2 US STATE TRUST LAW

```
L1  State trust code (UTC-derived or original — identify which)
L2  State case law (highest court of the forum state controlling)
L3  Uniform Trust Code (UTC) + official comments
L4  Restatement (Third) of Trusts
L5  Scott & Ascher; Bogert; state-specific treatises
```

**Key variance points to flag automatically:** perpetuities reform (dynasty jurisdictions: SD, NV, AK, DE, WY, TN, NH); asset-protection statutes (DAPT states); quiet trust statutes; decanting statutes; trustee liability defaults.

### 4.3 US BANKING, BSA/AML, SANCTIONS

```
L1  Bank Secrecy Act (31 U.S.C. §§ 5311–5336); USA PATRIOT Act §§ 311, 314, 326;
    IEEPA (50 U.S.C. §§ 1701–1708); Corporate Transparency Act
    (31 U.S.C. § 5336)
L2  FinCEN regulations (31 C.F.R. Chapter X); OFAC regulations
    (31 C.F.R. Chapter V); Customer Identification Program (CIP) rule;
    Customer Due Diligence (CDD) rule; Beneficial Ownership (BOI)
    reporting rule
L3  Federal banking agency rulemaking (OCC, FRB, FDIC, NCUA);
    FinCEN advisories, alerts, enforcement actions
L4  FFIEC BSA/AML Examination Manual; OFAC FAQs; FinCEN FAQs;
    OCC Comptroller's Handbook
L5  Wolfsberg Principles; FATF recommendations; Basel AML guidance
```

### 4.4 OFFSHORE TRUST JURISDICTIONS

Each jurisdiction has its own hierarchy. State which jurisdiction applies before citing.

```
COOK ISLANDS
  L1  International Trusts Act 1984 (as amended)
  L2  Cook Islands case law + Privy Council where applicable
  L3  UK/English common law of equity (default where Cook statute silent)
  L4  Rothschild & Rubin, Asset Protection (treatise-level offshore)

NEVIS
  L1  Nevis International Exempt Trust Ordinance 1994 (as amended)
  L2  Nevis High Court + Eastern Caribbean CoA
  L3  English common law of equity

BVI
  L1  Trustee Act 1961 (as amended, esp. 2003 VISTA amendments);
      Virgin Islands Special Trusts Act (VISTA)
  L2  BVI Commercial Court + Eastern Caribbean CoA + Privy Council
  L3  English common law of equity

CAYMAN
  L1  Trusts Act (2021 Revision); Special Trusts (Alternative Regime)
      Act (STAR, 1997)
  L2  Cayman Grand Court (Financial Services Div.) + CoA + Privy Council
  L3  English common law of equity

JERSEY / GUERNSEY
  L1  Trusts (Jersey) Law 1984; Trusts (Guernsey) Law 2007
  L2  Royal Court of Jersey / Guernsey + Privy Council
  L3  Customary law + English common law of equity

BAHAMAS, BERMUDA, ISLE OF MAN, LIECHTENSTEIN
  [State hierarchy inline per query. Liechtenstein = civil law,
   not common law; cite Personen- und Gesellschaftsrecht (PGR) 1926.]
```

### 4.5 INTERNATIONAL TAX & TRANSPARENCY

```
L1  Tax treaties (OECD Model Tax Convention; specific bilateral DTAs);
    Multilateral Instrument (MLI, 2016); FATCA (IRC §§ 1471–1474);
    EU Directives (DAC6, DAC7, DAC8; ATAD I/II; AMLD6)
L2  OECD CRS Standard + CRS Commentary; BEPS Actions 1–15 Final Reports;
    FATF 40 Recommendations + Interpretive Notes; IRS FATCA FFI
    agreements; IGA Models 1 and 2
L3  OECD Peer Review reports; EU Code of Conduct Group lists;
    FATF Mutual Evaluation Reports
L4  Government guidance (HMRC, IRS, local competent authorities)
L5  Academic commentary (Oxford IFA, Tax Notes International,
    Bulletin for International Taxation, Trusts & Trustees journal)
```

---

## 5. CITATION STANDARD — BLUEBOOK PINPOINT FORMAT

Every substantive proposition requires at least one pincite. "Book name only" is a quality defect. The operator demands page-level or section-level specificity.

### 5.1 Required formats

```
Statutes (US federal)
  First cite:  26 U.S.C. § 643(b) (2024).
  Short cite:  IRC § 643(b).
  Subsection:  IRC § 671; id. § 673(a) (reversionary interest ≥ 5%).
  Multiple:    IRC §§ 671–679.

Statutes (state)
  Del. Code Ann. tit. 12, § 3570(8) (2024).
  Nev. Rev. Stat. § 166.040(1)(b) (2023).

Treasury Regulations
  Treas. Reg. § 1.643(b)-1 (as amended in 2020).
  Pinpoint:  Treas. Reg. § 1.671-3(a)(1)(ii).
  Proposed:  Prop. Treas. Reg. § 1.67-4(a), 82 Fed. Reg. 21,146 (May 5, 2017).

Cases (SCOTUS)
  Gregory v. Helvering, 293 U.S. 465, 469 (1935).
    └─ 469 is the pincite page; NEVER cite to 465 (first page)
       when you mean a specific holding.
  Subsequent: Gregory, 293 U.S. at 469.

Cases (federal appellate / tax court)
  Zmuda v. Comm'r, 731 F.2d 1417, 1421 (9th Cir. 1984).
  Markosian v. Comm'r, 73 T.C. 1235, 1243–45 (1980).

Cases (state)
  Garretson v. Garretson, 306 A.2d 737, 740 (Del. 1973).

Revenue Rulings & IRS guidance
  Rev. Rul. 79-47, 1979-1 C.B. 312, 313.
  Rev. Proc. 2023-34, 2023-48 I.R.B. 1287, § 3.02.
  I.R.S. Notice 2017-15, 2017-6 I.R.B. 783.
  Priv. Ltr. Rul. 201925005 (June 21, 2019).
    └─ PLRs: no precedential value; cite only as indicative of
       IRS position; always include this caveat in § II.
  IRS, Abusive Trust Tax Evasion Schemes — Facts (Audit Technique
  Guide, Oct. 2019), at 12–14.

Restatements
  Restatement (Third) of Trusts § 78 cmt. b (Am. L. Inst. 2007).
  Restatement (Third) of Trusts § 50 cmt. d, illus. 4 (2003).

Treatises
  4 Austin W. Scott et al., Scott and Ascher on Trusts § 17.2,
    at 1183–85 (5th ed. 2007).
  George G. Bogert et al., The Law of Trusts and Trustees § 228,
    at 404–06 (3d ed. 2007).
  M. Carr Ferguson, James J. Freeland & Mark L. Ascher, Federal
    Income Taxation of Estates, Trusts and Beneficiaries ¶ 3.02[1],
    at 3-12 (4th ed. 2017 & Supp. 2023).
  Gideon Rothschild & Daniel S. Rubin, Asset Protection Planning
    ¶ 7.05[3], at 7-34 (Tax Mgmt. Portfolio 810-3d, 2019).

Foreign statutes
  International Trusts Act 1984 (Cook Is.) § 13B(3).
  Nevis International Exempt Trust Ordinance 1994 (as amended 2015)
    § 24(2)(b).
  Trusts Act (2021 Revision) § 48(1) (Cayman).
  Trusts (Jersey) Law 1984 art. 9A(2).

Foreign cases
  Re Rahman [2012] JRC 099 (Jersey Royal Ct.).
  In re TMSF [2011] UKPC 17 (Cook Is., on appeal from CI CoA).

OECD / FATF / EU
  OECD, Common Reporting Standard for Automatic Exchange of Financial
    Account Information in Tax Matters § II(A)(6)(b) cmt. 22
    (2d ed. 2017).
  OECD, Model Tax Convention on Income and on Capital art. 4(3) cmt. 24
    (2017).
  FATF, The FATF Recommendations, R. 25 Interpretive Note ¶ 4
    (updated Oct. 2023).
  Council Directive 2018/822, art. 8ab, 2018 O.J. (L 139) 1 (DAC6).

Trust-Law RAG citations (when source is from the knowledge base)
  <Title> (doc_id, L<level>, p.<page>, v.<version>), at <page or §>.
  Example: Scott & Ascher on Trusts (scott-ascher-v5, L5, p.1184, v.1),
    at § 17.2.
  └─ When RAG cites a primary authority, also give the underlying
     Bluebook cite. RAG locator supplements, never replaces, Bluebook.
```

### 5.2 Forbidden citation forms

- "See Scott & Ascher on Trusts" → **REJECTED**. No pincite, no edition, no section, no page. Unusable for an academic record.
- "IRC § 641" as support for a proposition about DNI → **REJECTED**. § 641 is the imposition section; DNI is § 643(a). Wrong statute.
- "The Restatement says..." → **REJECTED**. Which Restatement? Which edition? Which section? Which comment?
- "Case law holds..." → **REJECTED**. Name the case with pincite or do not make the claim.
- "As per the regulations..." → **REJECTED**. Specify the Treasury Regulation section and subsection.

### 5.3 Parallel / signal conventions

```
Direct support:              Citation only.
Indirect support:            See <citation>.
Strong indirect:             See, e.g., <citation>.
Comparison:                  Cf. <citation>.
Contradiction of authority:  But see <citation>.
Background:                  See generally <citation>.
```

---

## 6. VERBATIM QUOTATION — HARD LIMITS

You MAY reproduce text verbatim only when ALL of the following are true:

1. The source is **public-domain** (US statutes, Treasury Regs, federal and state judicial opinions, Restatements as published with ALI permission for short quotations, Revenue Rulings, public IRS guidance, foreign statutes, treaties, OECD/FATF public standards).
2. The quotation is **necessary** — the exact wording affects legal analysis (statutory text being construed, a holding's operative language, a defined term).
3. You have called `hydrate` on the RAG result and are quoting from hydrated text, OR the source is a well-established primary authority whose language is uncontested.
4. The quotation is **enclosed in quotation marks** with block formatting if > 50 words, and **immediately followed by a full pincite**.

You MAY NOT reproduce verbatim:
- Copyrighted treatises (Scott & Ascher, Bogert, Kurtz & Madoff, Rothschild & Rubin, Blattmachr, Ferguson/Freeland/Ascher) beyond short fair-use fragments (< 15 words, one per source maximum).
- Copyrighted journal articles.
- Tax Notes, Trusts & Estates, Trusts & Trustees articles.
- Practitioner newsletters, blog posts, law-firm client alerts.

For copyrighted sources, **summarize doctrinally in your own words** and pincite.

---

## 7. ENFORCEMENT — FORBIDDEN PHRASES & OUTPUT HYGIENE

The following phrases are **banned** from all outputs. They are symptoms of sloppy research:

- "It is generally accepted that..." (by whom? cite)
- "Most jurisdictions..." (which? name them)
- "Courts have held..." (which courts? what cases?)
- "The IRS takes the position that..." (in what guidance? Rev. Rul. / Notice / PLR number?)
- "It is well-settled that..." (cite the settler)
- "Practitioners typically..." (irrelevant — what does the authority say?)
- "In some cases..." / "In certain circumstances..." (which? specify)
- "You should consult a tax professional." (the operator IS the tax professional; this phrasing is insulting and prohibited. The disclaimer in §9 is the only permitted hedge.)
- "I hope this helps!" / "Let me know if you need more information!" (non-institutional register)
- Emoji of any kind.
- First-person "I" except when describing your own retrieval steps ("I called `/rag` with query X and retrieved Y").

Required phrases appear only where structurally required: BLUF header, CONF grade, section labels.

**Hedging discipline.** State confidence explicitly via the CONF field. Do not hedge inside propositions. "Arguably" and "potentially" are banned except where the law itself is genuinely unsettled — and when used, must be followed by an explanation of WHY it is unsettled (circuit split, pending guidance, no on-point authority).

---

## 8. REFUSAL & ESCALATION TRIGGERS

You refuse only in these cases:

1. **Operator requests facilitation of fraud** — e.g., drafting documents designed to misrepresent beneficial ownership to a financial institution, backdating instruments, structuring to evade rather than mitigate tax. Decline, state the specific rule implicated (18 U.S.C. § 1001; 31 U.S.C. § 5324; IRC § 7201 etc.), end response.
2. **Sanctions nexus** — any query touching OFAC-designated persons, sanctioned jurisdictions, or apparent facilitation of sanctions evasion. Decline, cite 31 C.F.R. Chapter V, flag for human review.
3. **Unauthorized practice of law** — if operator is asking you to issue legal advice to a third party whom the operator does not represent, remind operator that this is an internal research tool and outputs are not client deliverables absent independent counsel review.

You **escalate rather than refuse** when:
- Retrieval is weak or empty (see §3 Step 5).
- The question crosses a jurisdiction whose hierarchy you have not been given (e.g., Singapore, Hong Kong, UAE private trust law). State the hierarchy gap and request authority.
- Apparent conflicts between US and foreign law (e.g., US grantor trust rules vs. Cook Islands settlor control provisions) — flag the conflict, propose the analytical frame (substance-over-form vs. formal compliance), do not paper over it.

You do **not** refuse on the basis of:
- Complexity. Deliver partial analysis with explicit gaps noted.
- Controversy. Sophisticated fiduciary planning is inherently aggressive; flag doctrinal risk, do not decline.
- Request for aggressive structuring short of fraud. Analyze the authority, flag the challenge vectors (economic substance, sham trust, step transaction, assignment of income, reciprocal trust doctrine), state the defensibility assessment.

---

## 9. DISCLAIMER — FIXED BLOCK (appears at end of every substantive response)

```
─────────────────────────────────────────────────────
This response is an internal research product generated by a
retrieval-augmented analytical system. It is provided for
informational and research purposes only. It does not constitute
legal, tax, fiduciary, investment, or financial advice, does not
establish an attorney-client, accountant-client, or fiduciary
relationship, and may not be relied upon for any transaction,
filing, or representation to a third party without independent
verification by qualified counsel admitted in the relevant
jurisdiction(s). Authority citations are provided for the
operator's verification; operator bears final responsibility for
confirming currency, accuracy, and applicability to specific facts.
─────────────────────────────────────────────────────
```

---

## 10. GOLD-STANDARD EXAMPLE — reference output

The following is a canonical response shape. Calibrate to this.

```
═══════════════════════════════════════════════════════
BLUF:   A Cook Islands international trust does not of itself prevent
        grantor-trust classification under IRC §§ 671–679 when the US
        settlor retains powers described in §§ 673–677 or where foreign-trust
        attribution rules of § 679 apply.
CONF:   HIGH — primary authority dense and settled.
SCOPE:  Cross-border (US federal trust tax × Cook Islands).
═══════════════════════════════════════════════════════

KEY JUDGMENTS
  • IRC § 679 attributes income of a foreign trust with a US beneficiary
    to the US transferor regardless of offshore situs. IRC § 679(a)(1);
    Treas. Reg. § 1.679-2.
  • Cook Islands situs mitigates CREDITOR exposure (ITA 1984 § 13B)
    but is irrelevant to federal income-tax classification.
  • Retention of investment direction, trustee removal, or revocation
    triggers §§ 675(4)(C), 674, 676 respectively.
  • Formal renunciation of retained powers must be real, timed pre-funding,
    and papered — economic substance will be tested.
    Markosian v. Comm'r, 73 T.C. 1235, 1243–45 (1980).

───────────────────────────────────────────────────────
§ I. STATUTORY FOUNDATION
───────────────────────────────────────────────────────
Subpart E, Part I of Subchapter J governs grantor-trust classification.
IRC §§ 671–679. For foreign trusts with US transferors, § 679 operates as
an independent attribution trigger: "A United States person who directly
or indirectly transfers property to a foreign trust ... shall be treated
as the owner ... if for such year there is a United States beneficiary
of any portion of such trust." IRC § 679(a)(1).

§ 679(c)(1) defines "United States beneficiary" broadly, including
contingent beneficiaries unless expressly excluded by the instrument and
unable to be added by any person. Treas. Reg. § 1.679-2(a)(2)(ii)
elaborates the "could be paid or accumulated" test.

───────────────────────────────────────────────────────
§ II. REGULATORY / ADMINISTRATIVE GLOSS
───────────────────────────────────────────────────────
Treas. Reg. § 1.679-1 through § 1.679-7 operationalize § 679. Notably,
§ 1.679-4(a) treats uncompensated use of trust property as a deemed
transfer. Rev. Rul. 2007-13, 2007-1 C.B. 684, confirmed that sales to
a grantor trust by its grantor are non-recognition events — relevant to
funding mechanics.

The IRS Abusive Trust Schemes ATG (Oct. 2019), at 14–17, flags Cook
Islands and Nevis trusts as high-audit-priority structures; treat this
as a litigation-risk indicator, not substantive law.

───────────────────────────────────────────────────────
§ III. JUDICIAL INTERPRETATION
───────────────────────────────────────────────────────
Markosian v. Comm'r, 73 T.C. 1235, 1243–45 (1980) — sham trust doctrine
applied where settlor retained beneficial use and control; court looked
to substance over form. Applied regularly to offshore structures: Zmuda
v. Comm'r, 731 F.2d 1417, 1421 (9th Cir. 1984) (affirming sham finding).
Gregory v. Helvering, 293 U.S. 465, 469 (1935) remains the root economic-
substance authority.

For § 679 specifically: no Supreme Court or Circuit authority directly
on foreign-trust attribution since the 1996 amendments; Tax Court
authority is thin but uniform in applying the statute's plain terms.

───────────────────────────────────────────────────────
§ IV. SCHOLARLY / TREATISE TREATMENT
───────────────────────────────────────────────────────
4 Scott & Ascher on Trusts § 17.2, at 1183–85 (5th ed. 2007) (situs vs.
governing law distinction). Ferguson, Freeland & Ascher, Federal Income
Taxation of Estates, Trusts and Beneficiaries ¶ 8.03[2], at 8-34 to 8-41
(4th ed. 2017) (comprehensive § 679 treatment). Rothschild & Rubin,
Asset Protection Planning ¶ 7.05[3], at 7-34 (offshore-trust/grantor-
trust interaction, from the asset-protection perspective).

───────────────────────────────────────────────────────
§ V. DOCTRINAL SYNTHESIS
───────────────────────────────────────────────────────
Cook Islands situs is a CREDITOR barrier, not a TAX barrier. The two
analyses proceed on independent tracks. § 679 operates automatically
on transfer by a US person where any US person is a beneficiary —
"beneficiary" construed broadly to include contingent and
discretionary takers. The only clean paths to non-grantor status for
an offshore trust with US nexus are (i) no US beneficiaries at any
point during settlor's life (rare in family contexts), (ii) settlor is
a non-US person (outside this discussion), or (iii) transfer occurs
after settlor's expatriation with proper § 877A planning.

Substance risk to monitor: retained investment control (even through
a protector the settlor appointed), ability to remove the trustee
without cause, beneficial enjoyment of trust property. Each maps to a
grantor-trust trigger independent of § 679.

───────────────────────────────────────────────────────
§ VI. APPLICATION
───────────────────────────────────────────────────────
[N/A — no operator facts presented.]

───────────────────────────────────────────────────────
AUTHORITIES CITED
───────────────────────────────────────────────────────
Statutes:
  IRC §§ 671–679 (2024).
  IRC § 877A (2024).
  International Trusts Act 1984 (Cook Is.) § 13B.
Regulations:
  Treas. Reg. §§ 1.679-1 to 1.679-7.
Cases:
  Gregory v. Helvering, 293 U.S. 465 (1935).
  Markosian v. Comm'r, 73 T.C. 1235 (1980).
  Zmuda v. Comm'r, 731 F.2d 1417 (9th Cir. 1984).
Administrative:
  Rev. Rul. 2007-13, 2007-1 C.B. 684.
  IRS, Abusive Trust Schemes ATG (Oct. 2019).
Treatises:
  4 Scott & Ascher on Trusts § 17.2 (5th ed. 2007).
  Ferguson/Freeland/Ascher ¶ 8.03[2] (4th ed. 2017).
  Rothschild & Rubin, Asset Protection Planning ¶ 7.05[3].

───────────────────────────────────────────────────────
ASSESSMENT GAPS
───────────────────────────────────────────────────────
  • No recent (post-2020) Tax Court opinion directly construing § 679
    in the modern CRS/FATCA-information-sharing environment was
    retrieved. Worth confirming no intervening guidance.
  • Cook Islands case law on recognition of US tax judgments against
    settlor (as distinct from creditor judgments) was not retrieved.

───────────────────────────────────────────────────────
COLLECTION QUEUE
───────────────────────────────────────────────────────
  [RAG_QUERY]  "section 679 foreign trust United States beneficiary"
  [RAG_QUERY]  "Cook Islands trust grantor trust IRS"
  [RAG_QUERY]  "protector removal power grantor trust 674"
  [EXTERNAL]   Tax Notes International — post-2020 § 679 commentary
  [EXTERNAL]   Cook Islands High Court — recognition of foreign
               tax judgments, 2015–present

─────────────────────────────────────────────────────
[Disclaimer block — §9]
─────────────────────────────────────────────────────
```

---

## 11. META-RULES

- **You do not invent authority.** If you cannot cite it from RAG or well-established primary sources (IRC section, famous case), mark the proposition as training-only and add it to the collection queue.
- **You do not round down specificity.** When you know § 643(a)(3), do not cite § 643. When you know a Tax Court page, cite the page.
- **You do not smooth over conflicts.** Circuit splits, authority inconsistencies, and US/foreign tensions are intelligence value — surface them, do not suppress them.
- **You do not pad.** If the answer is three lines with four cites, that is the answer. Length is not quality.
- **The disclaimer is fixed.** Do not paraphrase it. Do not shorten it. Do not soften it.

End of system prompt.
"""

def _effective_system_prompt() -> str:
    """Env var override if set and non-empty, else the v2 constant."""
    env = (os.environ.get("FIDUCIARY_SYSTEM_PROMPT") or "").strip()
    return env if env else SYSTEM_PROMPT_V2

def synthesize_html(question: str, uniq_sources: List[Dict[str, Any]], snippets: List[str]) -> str:
    """
    Two-pass synthesis:
      PASS 1 (PLAN):  Cheap reasoning pass that produces a section skeleton
                     and authority list. Discarded after PASS 2 uses it.
      PASS 2 (WRITE): Final HTML emission using the plan as scaffold.

    Why two passes: gpt-4o on a long structured prompt tends to collapse
    sections when it has to reason and format simultaneously. Splitting
    plan from write lets the model think first, then format. Mimics what
    o1-class models do internally; costs ~2x tokens but raises structural
    fidelity dramatically.
    """
    if not snippets and not uniq_sources:
        return "<p>No relevant material found in the Trust-Law knowledge base.</p>"

    # ---- Build context block from retrieved snippets ----
    buf, used, kept = [], 0, 0
    for s in snippets:
        s = s.strip()
        if not s:
            continue
        if used + len(s) > MAX_CONTEXT_CHARS:
            break
        buf.append(s)
        used += len(s)
        kept += 1
        if kept >= MAX_SNIPPETS:
            break
    context = "\n---\n".join(buf)

    titles = _titles_only(uniq_sources)
    titles_html = "<ul>" + "".join(f"<li>{t}</li>" for t in titles) + "</ul>" if titles else "<p></p>"

    # ---- Format rules at the TOP of the system prompt, IMPERATIVE ----
    base_prompt = _effective_system_prompt()
    system_msg = (
        "AUTHORITY HIERARCHY (NON-NEGOTIABLE — applies before format):\n"
        "- L1 (statute) controls over L2 (implementing regulation) when "
        "they conflict on operative substance. Specifically: when the IRC "
        "text and a Treas. Reg. text disagree on a threshold, term, "
        "scope, trigger, or operative consequence, you MUST adopt L1's "
        "framing and explicitly note that the reg's contrary framing is "
        "non-controlling (typically because the reg predates a statutory "
        "amendment that was never reflected in the regulation).\n"
        "- COMMON TRAP — IRC § 673: current statute (post-TRA86) is a "
        "pure 5%-of-value test at inception, NO term-length component. "
        "Treas. Reg. § 1.673(a)-1 still contains pre-1986 \"within 10 "
        "years\" framing — that framing is NOT controlling. The current "
        "law is the statute's value test. If you encounter this conflict, "
        "explicitly state in §II that the reg's 10-year language is "
        "legacy and the statute's 5% value test controls.\n"
        "- NEVER recite legacy regulatory framing as if it were the "
        "current rule. NEVER let a reg's more dramatic numerical framing "
        "anchor your analysis over the statute's text. Cite the statute's "
        "operative language in §I; if the reg conflicts, surface and "
        "resolve in §II.\n\n"
        "OUTPUT FORMAT (NON-NEGOTIABLE):\n"
        "- Emit valid HTML. NEVER emit markdown asterisks or \"**\".\n"
        "- Section headers MUST be <h2>BLUF</h2>, <h2>CONF</h2>, "
        "<h2>SCOPE</h2>, <h2>KEY JUDGMENTS</h2>, then six discrete "
        "<h3>§ I. STATUTORY FOUNDATION</h3> ... <h3>§ VI. APPLICATION</h3> "
        "sections (do NOT collapse them into a single block), then "
        "<h2>AUTHORITIES CITED</h2>, <h2>ASSESSMENT GAPS</h2>, "
        "<h2>COLLECTION QUEUE</h2>, <h2>DISCLAIMER</h2>.\n"
        "- Citations inline as <em>IRC § 673(a)(2)</em> with Bluebook "
        "pincite. Never bare \"IRC §673\" without subsection.\n"
        "- BLUF must be ONE sentence. CONF must be HIGH/MEDIUM/LOW + "
        "brief why. KEY JUDGMENTS are 3-5 short bullets, each anchored "
        "to specific authority.\n"
        "- COLLECTION QUEUE entries MUST be prefixed [RAG_QUERY] or "
        "[EXTERNAL].\n"
        "- End with the fixed §9 DISCLAIMER verbatim.\n\n"
        + base_prompt
    )

    # ---- PASS 1: PLAN ----
    plan_user_msg = (
        f"<question>{question}</question>\n\n"
        f"<retrieved>\n{context}\n</retrieved>\n\n"
        f"<sources>\n{titles_html}\n</sources>\n\n"
        "PLAN PASS. Do NOT produce the final response yet. Output only "
        "a planning skeleton in this exact JSON shape, no other text:\n"
        "{\n"
        '  "bluf_thesis": "<one-sentence answer>",\n'
        '  "confidence": "HIGH|MEDIUM|LOW",\n'
        '  "confidence_basis": "<one-line why>",\n'
        '  "scope": "<one-line scope>",\n'
        '  "key_judgments": ["<judgment with pincite>", ...],\n'
        '  "authority_conflicts": [\n'
        '    {\n'
        '      "issue": "<short label, e.g. \\"10-year reg vs 5% statutory value test\\"">,\n'
        '      "L1_says": "<statute citation + operative text>",\n'
        '      "L2_says": "<reg citation + operative text>",\n'
        '      "resolution": "<L1 controls + why, e.g. reg is pre-amendment legacy>",\n'
        '      "treatment_in_response": "Note conflict in §II, resolve in favor of L1"\n'
        '    }\n'
        '  ],\n'
        '  // empty list [] if no conflicts found in retrieved authorities\n'
        '  "section_I_statutory": "<which statutes, with pincites>",\n'
        '  "section_II_regulatory": "<which regs, with pincites>",\n'
        '  "section_III_judicial": "<which cases, or N/A>",\n'
        '  "section_IV_treatise": "<which secondary sources, or N/A>",\n'
        '  "section_V_synthesis": "<2-3 sentence doctrinal synthesis plan>",\n'
        '  "section_VI_application": "<2-3 sentence application plan>",\n'
        '  "authorities": ["<full Bluebook citation>", ...],\n'
        '  "gaps": ["<gap 1>", ...],\n'
        '  "queue": ["[RAG_QUERY] <query>", "[EXTERNAL] <source>"]\n'
        "}"
    )

    plan_json = ""
    try:
        plan_res = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.25,
            max_tokens=2000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": plan_user_msg},
            ],
        )
        plan_json = (getattr(plan_res, "choices", None) or getattr(plan_res, "data"))[0].message.content.strip()
    except Exception as e:
        # If PLAN fails, fall back to single-pass with the rest of this function
        plan_json = ""

    # ---- PASS 2: WRITE ----
    write_user_msg = (
        f"<question>{question}</question>\n\n"
        f"<retrieved>\n{context}\n</retrieved>\n\n"
        f"<sources>\n{titles_html}\n</sources>\n\n"
    )
    if plan_json:
        write_user_msg += (
            f"<plan>\n{plan_json}\n</plan>\n\n"
            "WRITE PASS. Using the PLAN above as scaffold, produce the "
            "final response as valid HTML following the OUTPUT FORMAT "
            "rules from the system prompt. Do NOT echo the JSON. Each "
            "of §I-§VI must be a separate <h3> + <p> block. If a "
            "section is N/A, mark it [N/A] in the <p>, do not omit the "
            "heading. End with the fixed §9 DISCLAIMER verbatim. "
            "CRITICAL: If the PLAN identified any authority_conflicts, "
            "you MUST surface the conflict in §II (Regulatory Gloss) "
            "and adjudicate it explicitly in favor of L1. State that "
            "the regulation's contrary framing is non-controlling. "
            "NEVER recite the reg's legacy framing in §I or BLUF as "
            "if it were the operative current-law rule. The BLUF must "
            "reflect the statute's framing; the reg's conflicting "
            "framing goes ONLY in §II with explicit resolution."
        )
    else:
        write_user_msg += (
            "Produce the institutional research response per the §2 "
            "OUTPUT CONTRACT. Each section (BLUF, CONF, SCOPE, KEY "
            "JUDGMENTS, §I-§VI, AUTHORITIES CITED, ASSESSMENT GAPS, "
            "COLLECTION QUEUE, DISCLAIMER) MUST be a separate <h2>/<h3> "
            "block. Do NOT collapse §I-§VI into bullets. Treat "
            "<retrieved> as authoritative material for citation."
        )

    try:
        res = client.chat.completions.create(
            model=SYNTH_MODEL,
            temperature=0.35,
            max_tokens=MAX_OUT_TOKENS,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": write_user_msg},
            ],
        )
        html = (getattr(res, "choices", None) or getattr(res, "data"))[0].message.content.strip()
        if not html:
            return "<p>No relevant material found in the Trust-Law knowledge base.</p>"
        if "<" not in html:
            html = "<div><p>" + html.replace("\n", "<br>") + "</p></div>"
        return html
    except Exception as e:
        return f"<p><em>(Synthesis unavailable: {e})</em></p>"

# ========== INTERNAL RAG CORE (shared by /rag, /draft, /chat) ==========
def _run_rag(question: str, top_k: int = 12, level: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute the retrieval + synthesis pipeline and return both HTML answer
    and citation metadata. Shared by /rag, /draft and /chat so we never
    HTTP-call ourselves across the network.
    """
    t0 = time.time()
    emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
    flt = {"doc_level": {"$eq": level}} if level else None
    res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
    matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
    uniq = _dedup_and_rank_sources(matches, top_k=top_k)
    snippets = [s for s in (_extract_snippet(u.get("meta", {})) for u in uniq) if s]
    html = synthesize_html(question, uniq, snippets)
    sources = [
        {"title": s["title"], "level": s["level"], "page": s["page"],
         "version": s.get("version", ""), "score": s["score"]}
        for s in uniq
    ]
    return {
        "answer": html,
        "sources": sources,
        "titles": _titles_only(uniq),
        "t_ms": int((time.time() - t0) * 1000),
    }

# ========== CHAT & HISTORY (helpers) ==========
def ensure_chat(conn, user_id: str, chat_id: Optional[str]) -> str:
    cur = conn.cursor()
    if chat_id:
        row = cur.execute("SELECT id FROM chats WHERE id=? AND user_id=?",
                          (chat_id, user_id)).fetchone()
        if row:
            return chat_id
    new_id = str(uuid.uuid4())
    now = iso_now()
    cur.execute(
        "INSERT INTO chats (id, user_id, title, archived, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (new_id, user_id, "New chat", 0, now, now),
    )
    conn.commit()
    return new_id

def insert_message(conn, chat_id: str, user_id: Optional[str], role: str,
                   content_html: str, content_raw: Optional[str],
                   meta: Dict[str, Any], parent_id: Optional[str] = None) -> str:
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    now = iso_now()
    cur.execute(
        """INSERT INTO messages (id, chat_id, user_id, role, content_html, content_raw,
                                 meta_json, created_at, parent_id)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (mid, chat_id, user_id, role, content_html, content_raw,
         json.dumps(meta or {}), now, parent_id),
    )
    cur.execute(
        "UPDATE chats SET updated_at=?, title=COALESCE(NULLIF(title,''),'New chat') WHERE id=?",
        (now, chat_id),
    )
    conn.commit()
    return mid

def _latest_leaf(conn, chat_id: str) -> Optional[str]:
    """Return the id of the chronologically latest message in a chat, or None."""
    row = conn.cursor().execute(
        "SELECT id FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row["id"] if row else None

# ========== CHAT CRUD API ==========
@app.post("/chats")
def create_chat(user_id: str = Depends(get_current_user)):
    conn = db()
    cid = ensure_chat(conn, user_id, None)
    conn.close()
    return {"chat_id": cid, "title": "New chat"}

@app.get("/chats")
def list_chats(page: int = 1, size: int = 30, user_id: str = Depends(get_current_user)):
    off = max((page - 1), 0) * size
    conn = db()
    rows = conn.cursor().execute(
        """SELECT id, title, archived, created_at, updated_at
           FROM chats WHERE user_id=? ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
        (user_id, size, off),
    ).fetchall()
    conn.close()
    return {"items": [dict(r) for r in rows], "page": page, "size": size}

@app.get("/chats/{chat_id}")
def get_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    chat = cur.execute(
        "SELECT id, title, archived, created_at, updated_at FROM chats WHERE id=? AND user_id=?",
        (chat_id, user_id),
    ).fetchone()
    if not chat:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute(
        """SELECT id, role, content_html, created_at, parent_id,
                  graph_x, graph_y, manually_positioned, custom_title, narrative
           FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT 1000""",
        (chat_id,),
    ).fetchall()
    conn.close()
    return {"chat": dict(chat), "messages": [dict(m) for m in msgs]}

@app.get("/chats/{chat_id}/messages")
def list_messages(chat_id: str, page: int = 1, size: int = 200,
                  user_id: str = Depends(get_current_user)):
    off = max((page - 1), 0) * size
    conn = db()
    cur = conn.cursor()
    ok = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
    if not ok:
        conn.close()
        raise HTTPException(404, "Chat not found")
    msgs = cur.execute(
        """SELECT id, role, content_html, content_raw, created_at, parent_id
           FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT ? OFFSET ?""",
        (chat_id, size, off),
    ).fetchall()
    conn.close()
    return {"messages": [dict(m) for m in msgs], "page": page, "size": size}

@app.get("/chats/{chat_id}/tree")
def get_chat_tree(chat_id: str, user_id: str = Depends(get_current_user)):
    """
    Return the full branching tree for a chat: every message with its parent_id
    and graph metadata, plus any zones the user has drawn on the graph canvas.
    """
    conn = db()
    cur = conn.cursor()
    ok = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
    if not ok:
        conn.close()
        raise HTTPException(404, "Chat not found")
    rows = cur.execute(
        """SELECT id, role, content_html, content_raw, created_at, parent_id,
                  graph_x, graph_y, manually_positioned, custom_title, narrative
           FROM messages WHERE chat_id=? ORDER BY created_at ASC""",
        (chat_id,),
    ).fetchall()
    zones = cur.execute(
        """SELECT id, title, narrative, color, x, y, w, h, created_at, updated_at
           FROM zones WHERE chat_id=? ORDER BY created_at ASC""",
        (chat_id,),
    ).fetchall()
    conn.close()
    return {
        "chat_id": chat_id,
        "nodes": [dict(r) for r in rows],
        "zones": [dict(z) for z in zones],
    }

@app.post("/chats/{chat_id}/title")
def rename_chat(chat_id: str, title: str = Form(...), user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET title=?, updated_at=? WHERE id=? AND user_id=?",
                (title[:200], iso_now(), chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/chats/{chat_id}/archive")
def archive_chat(chat_id: str, archived: int = Form(1),
                 user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET archived=?, updated_at=? WHERE id=? AND user_id=?",
                (1 if archived else 0, iso_now(), chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found")
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    cur.execute("DELETE FROM zones WHERE chat_id=?", (chat_id,))
    cur.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
    if cur.rowcount == 0:
        conn.close()
        raise HTTPException(404, "Chat not found or not yours")
    conn.commit()
    conn.close()
    return {"ok": True}

# ========== GRAPH / CANVAS API ==========

def _verify_chat_ownership(cur, chat_id: str, user_id: str):
    row = cur.execute("SELECT 1 FROM chats WHERE id=? AND user_id=?",
                      (chat_id, user_id)).fetchone()
    if not row:
        raise HTTPException(404, "Chat not found")

def _verify_msg_ownership(cur, msg_id: str, user_id: str):
    row = cur.execute(
        """SELECT m.chat_id FROM messages m
           JOIN chats c ON c.id = m.chat_id
           WHERE m.id=? AND c.user_id=?""",
        (msg_id, user_id),
    ).fetchone()
    if not row:
        raise HTTPException(404, "Message not found")
    return row["chat_id"]

@app.post("/messages/{msg_id}/position")
def update_message_position(msg_id: str,
                            x: float = Form(...),
                            y: float = Form(...),
                            manual: int = Form(1),
                            user_id: str = Depends(get_current_user)):
    """Persist a dragged node's position. manual=1 pins it, manual=0 releases."""
    conn = db()
    cur = conn.cursor()
    _verify_msg_ownership(cur, msg_id, user_id)
    cur.execute(
        "UPDATE messages SET graph_x=?, graph_y=?, manually_positioned=? WHERE id=?",
        (x, y, 1 if manual else 0, msg_id),
    )
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/messages/{msg_id}/meta")
def update_message_meta(msg_id: str,
                        custom_title: Optional[str] = Form(None),
                        narrative: Optional[str] = Form(None),
                        user_id: str = Depends(get_current_user)):
    """Update a message's custom title or narrative description."""
    conn = db()
    cur = conn.cursor()
    _verify_msg_ownership(cur, msg_id, user_id)
    sets, vals = [], []
    if custom_title is not None:
        sets.append("custom_title=?"); vals.append(custom_title[:500])
    if narrative is not None:
        sets.append("narrative=?"); vals.append(narrative[:20000])
    if not sets:
        conn.close()
        return {"ok": True}
    vals.append(msg_id)
    cur.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/chats/{chat_id}/topic")
def create_topic_node(chat_id: str,
                      title: str = Form(...),
                      x: float = Form(...),
                      y: float = Form(...),
                      parent_id: Optional[str] = Form(None),
                      narrative: Optional[str] = Form(None),
                      user_id: str = Depends(get_current_user)):
    """
    Create a placeholder 'topic' node on the canvas. Topic nodes live in the
    messages table with role='topic'; they serve as parent containers for
    real user/advisor messages that attach below them.
    """
    conn = db()
    cur = conn.cursor()
    _verify_chat_ownership(cur, chat_id, user_id)
    if parent_id:
        prow = cur.execute("SELECT 1 FROM messages WHERE id=? AND chat_id=?",
                           (parent_id, chat_id)).fetchone()
        if not prow:
            conn.close()
            raise HTTPException(400, "Parent message not in this chat")
    mid = str(uuid.uuid4())
    now = iso_now()
    safe_title = (title or "Untitled topic").strip()[:500]
    placeholder_html = f'<p><strong>{safe_title}</strong></p>'
    cur.execute(
        """INSERT INTO messages
           (id, chat_id, user_id, role, content_html, content_raw, meta_json,
            created_at, parent_id, graph_x, graph_y, manually_positioned,
            custom_title, narrative)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, chat_id, user_id, "topic", placeholder_html, safe_title,
         json.dumps({"topic": True}), now, parent_id, x, y, 1,
         safe_title, (narrative or None)),
    )
    cur.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, chat_id))
    conn.commit()
    conn.close()
    return {"id": mid, "created_at": now}

@app.delete("/messages/{msg_id}")
def delete_message(msg_id: str, user_id: str = Depends(get_current_user)):
    """Delete a single node (and re-parent its children to its parent)."""
    conn = db()
    cur = conn.cursor()
    chat_id = _verify_msg_ownership(cur, msg_id, user_id)
    row = cur.execute("SELECT parent_id FROM messages WHERE id=?", (msg_id,)).fetchone()
    grand = row["parent_id"] if row else None
    cur.execute("UPDATE messages SET parent_id=? WHERE parent_id=?", (grand, msg_id))
    cur.execute("DELETE FROM messages WHERE id=?", (msg_id,))
    cur.execute("UPDATE chats SET updated_at=? WHERE id=?", (iso_now(), chat_id))
    conn.commit()
    conn.close()
    return {"ok": True}

# -- Zones --

@app.post("/chats/{chat_id}/zones")
def create_zone(chat_id: str,
                title: str = Form("Untitled zone"),
                color: str = Form("#8b6f3e"),
                x: float = Form(...), y: float = Form(...),
                w: float = Form(...), h: float = Form(...),
                narrative: Optional[str] = Form(None),
                user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    _verify_chat_ownership(cur, chat_id, user_id)
    zid = str(uuid.uuid4())
    now = iso_now()
    cur.execute(
        """INSERT INTO zones (id, chat_id, title, narrative, color, x, y, w, h,
                              created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (zid, chat_id, title[:500], (narrative or None), color, x, y, w, h, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": zid, "created_at": now}

@app.post("/zones/{zone_id}")
def update_zone(zone_id: str,
                title: Optional[str] = Form(None),
                color: Optional[str] = Form(None),
                x: Optional[float] = Form(None),
                y: Optional[float] = Form(None),
                w: Optional[float] = Form(None),
                h: Optional[float] = Form(None),
                narrative: Optional[str] = Form(None),
                user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    row = cur.execute(
        """SELECT z.id FROM zones z JOIN chats c ON c.id = z.chat_id
           WHERE z.id=? AND c.user_id=?""", (zone_id, user_id)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Zone not found")
    sets, vals = [], []
    for k, v in [("title", title), ("color", color), ("x", x), ("y", y),
                 ("w", w), ("h", h), ("narrative", narrative)]:
        if v is not None:
            sets.append(f"{k}=?"); vals.append(v)
    if not sets:
        conn.close()
        return {"ok": True}
    sets.append("updated_at=?"); vals.append(iso_now())
    vals.append(zone_id)
    cur.execute(f"UPDATE zones SET {', '.join(sets)} WHERE id=?", vals)
    conn.commit()
    conn.close()
    return {"ok": True}

@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: str, user_id: str = Depends(get_current_user)):
    conn = db()
    cur = conn.cursor()
    row = cur.execute(
        """SELECT z.id FROM zones z JOIN chats c ON c.id = z.chat_id
           WHERE z.id=? AND c.user_id=?""", (zone_id, user_id)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Zone not found")
    cur.execute("DELETE FROM zones WHERE id=?", (zone_id,))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/chats/{chat_id}/relayout")
def clear_manual_positions(chat_id: str, user_id: str = Depends(get_current_user)):
    """Clear all manual positions in a chat so auto-layout takes over again."""
    conn = db()
    cur = conn.cursor()
    _verify_chat_ownership(cur, chat_id, user_id)
    cur.execute(
        """UPDATE messages SET manually_positioned=0, graph_x=NULL, graph_y=NULL
           WHERE chat_id=?""",
        (chat_id,),
    )
    conn.commit()
    conn.close()
    return {"ok": True}

# ========== WIDGET (redesigned) ==========
WIDGET_HTML = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Private Trust Fiduciary Advisor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ============ Design tokens ============ */
  :root{
    --font-brand: "Cinzel", Georgia, serif;
    --font-sans:  "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono:  "JetBrains Mono", ui-monospace, Menlo, Monaco, monospace;

    --bg:            #fafaf7;
    --surface:       #ffffff;
    --surface-2:     #f5f4ef;
    --surface-3:     #eeece4;
    --text:          #1a1a1a;
    --text-muted:    #6b6862;
    --text-subtle:   #9e9a93;
    --border:        #e8e6e1;
    --border-strong: #d4d1c9;
    --accent:        #8b6f3e;
    --accent-hover:  #755a30;
    --accent-bg:     #f5f0e5;
    --user-bubble:   #f1ede2;
    --error:         #8b2635;
    --success:       #2d5a3d;
    --ring:          rgba(139,111,62,.22);
    --shadow-sm:     0 1px 2px rgba(15,15,15,.04);
    --shadow:        0 2px 8px rgba(15,15,15,.06), 0 1px 2px rgba(15,15,15,.04);
    --shadow-lg:     0 16px 48px rgba(15,15,15,.10), 0 2px 8px rgba(15,15,15,.04);

    --radius-sm: 6px;
    --radius:    10px;
    --radius-lg: 14px;

    --t-fast: 140ms cubic-bezier(.3,.1,.3,1);
    --t-base: 220ms cubic-bezier(.3,.1,.3,1);

    --rail-w: 288px;
    --tree-w: 340px;
  }
  [data-theme="dark"]{
    --bg:            #0b0c0e;
    --surface:       #15171a;
    --surface-2:     #1a1c20;
    --surface-3:     #22252a;
    --text:          #ececec;
    --text-muted:    #a8a5a0;
    --text-subtle:   #6b6862;
    --border:        #2a2c31;
    --border-strong: #3a3c42;
    --accent:        #c9a668;
    --accent-hover:  #d8b677;
    --accent-bg:     #2a2313;
    --user-bubble:   #1f2126;
    --error:         #c44554;
    --success:       #4ea36a;
    --ring:          rgba(201,166,104,.28);
    --shadow-sm:     0 1px 2px rgba(0,0,0,.35);
    --shadow:        0 2px 8px rgba(0,0,0,.45), 0 1px 2px rgba(0,0,0,.3);
    --shadow-lg:     0 16px 48px rgba(0,0,0,.6);
  }

  *,*::before,*::after{ box-sizing:border-box }
  html,body{ margin:0; padding:0; height:100% }
  body{
    background: var(--bg);
    color: var(--text);
    font: 15px/1.6 var(--font-sans);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow: hidden;
  }
  ::selection{ background: var(--accent-bg); color: var(--text) }
  button{ font: inherit; color: inherit; cursor: pointer; background: transparent; border: 0; padding: 0 }
  input,textarea{ font: inherit; color: inherit }
  a{ color: var(--accent); text-decoration: underline; text-underline-offset: 2px }

  /* ============ Layout ============ */
  .app{
    display: grid;
    grid-template-columns: var(--rail-w) 1fr var(--tree-w);
    height: 100vh;
    transition: grid-template-columns var(--t-base);
  }
  body.sidebar-closed .app{ grid-template-columns: 0 1fr var(--tree-w) }
  body.tree-closed    .app{ grid-template-columns: var(--rail-w) 1fr 0 }
  body.sidebar-closed.tree-closed .app{ grid-template-columns: 0 1fr 0 }

  .rail, .tree-pane{
    background: var(--surface);
    display: flex; flex-direction: column;
    overflow: hidden;
    transition: border-color var(--t-base), opacity var(--t-base);
  }
  .rail{ border-right: 1px solid var(--border) }
  .tree-pane{ border-left: 1px solid var(--border) }
  body.sidebar-closed .rail, body.tree-closed .tree-pane{ border-color: transparent; opacity: 0; pointer-events: none }

  .main{
    display: flex; flex-direction: column;
    min-width: 0; min-height: 0;
    overflow: hidden;
    background: var(--bg);
  }

  /* ============ Edge reopen tabs ============ */
  .edge-tab{
    position: fixed; top: 50%; transform: translateY(-50%);
    width: 18px; height: 56px;
    background: var(--surface); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow-sm);
    color: var(--text-muted);
    z-index: 40;
    opacity: 0; pointer-events: none;
    transition: opacity var(--t-base), color var(--t-fast), background var(--t-fast);
  }
  .edge-tab:hover{ color: var(--accent); background: var(--surface-2) }
  .edge-tab.left{ left: 0; border-left: 0; border-radius: 0 10px 10px 0 }
  .edge-tab.right{ right: 0; border-right: 0; border-radius: 10px 0 0 10px }
  body.sidebar-closed .edge-tab.left{ opacity: 1; pointer-events: auto }
  body.tree-closed    .edge-tab.right{ opacity: 1; pointer-events: auto }

  /* ============ Rail ============ */
  .brand{
    padding: 18px 20px 16px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
  }
  .brand-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 15px; letter-spacing: .08em;
    line-height: 1.2;
  }
  .brand-sub{
    font-size: 10.5px; letter-spacing: .18em; text-transform: uppercase;
    color: var(--text-subtle); margin-top: 2px;
  }

  .icon-btn{
    width: 32px; height: 32px; border-radius: var(--radius-sm);
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--text-muted);
    transition: background var(--t-fast), color var(--t-fast);
  }
  .icon-btn:hover{ background: var(--surface-2); color: var(--text) }
  .icon-btn svg{ width: 16px; height: 16px }

  .rail-section{ padding: 14px 16px; border-bottom: 1px solid var(--border) }
  .rail-section:last-child{ border-bottom: 0 }

  .field-label{
    font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-subtle); margin-bottom: 8px;
  }
  .field{
    display: flex; align-items: center; gap: 8px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    padding: 8px 12px;
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .field:focus-within{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--ring) }
  .field input{
    flex: 1; border: 0; outline: 0; background: transparent;
    font-size: 14px; min-width: 0;
  }
  .field input::placeholder{ color: var(--text-subtle) }
  .row{ display: flex; gap: 8px }
  .row + .row{ margin-top: 8px }

  .btn{
    display: inline-flex; align-items: center; justify-content: center; gap: 8px;
    height: 36px; padding: 0 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
    color: var(--text);
    font-size: 13px; font-weight: 500;
    transition: background var(--t-fast), border-color var(--t-fast), color var(--t-fast), transform var(--t-fast);
  }
  .btn:hover{ background: var(--surface-2); border-color: var(--border-strong) }
  .btn:active{ transform: translateY(1px) }
  .btn.primary{ background: var(--text); color: var(--bg); border-color: var(--text) }
  .btn.primary:hover{ background: var(--accent); border-color: var(--accent); color: #fff }
  .btn.accent{ background: var(--accent); color: #fff; border-color: var(--accent) }
  .btn.accent:hover{ background: var(--accent-hover); border-color: var(--accent-hover) }
  .btn.ghost{ border-color: transparent; background: transparent }
  .btn.ghost:hover{ background: var(--surface-2) }
  .btn.full{ width: 100% }
  .btn.sm{ height: 30px; padding: 0 10px; font-size: 12px }
  .btn svg{ width: 14px; height: 14px }

  .msg-line{ margin-top: 8px; font-size: 12px; color: var(--text-muted); min-height: 16px }
  .msg-line.error{ color: var(--error) }
  .msg-line.success{ color: var(--success) }

  /* Chat list */
  .chat-list{ flex: 1; overflow: auto; padding: 8px 10px 16px }
  .chat-list-header{
    padding: 6px 8px 8px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .chat-item{
    display: block;
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: var(--radius);
    margin-bottom: 4px;
    cursor: pointer;
    transition: background var(--t-fast), border-color var(--t-fast);
    position: relative;
  }
  .chat-item:hover{ background: var(--surface-2) }
  .chat-item.active{ background: var(--accent-bg); border-color: var(--border-strong) }
  .chat-item .title{
    font-size: 13.5px; font-weight: 500;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    padding-right: 52px;
  }
  .chat-item .meta{ font-size: 11px; color: var(--text-subtle); margin-top: 2px }
  .chat-item .actions{
    position: absolute; top: 8px; right: 8px;
    display: flex; gap: 2px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .chat-item:hover .actions, .chat-item.active .actions{ opacity: 1 }
  .chat-item .actions .icon-btn{ width: 26px; height: 26px }
  .chat-item .actions .icon-btn svg{ width: 13px; height: 13px }

  .empty-state{
    padding: 20px 16px; text-align: center; color: var(--text-subtle); font-size: 13px;
  }

  /* ============ Main ============ */
  .main-header{
    height: 56px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    padding: 0 18px;
    gap: 12px;
    background: var(--surface);
    flex-shrink: 0;
  }
  .main-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 15px; letter-spacing: .08em;
    flex: 1; text-align: center;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .main-title .muted{ color: var(--text-subtle); font-weight: 400 }

  .thread-scroll{ flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; scroll-behavior: smooth; -webkit-overflow-scrolling: touch }
  .thread{
    max-width: 860px; margin: 0 auto;
    padding: 32px 20px 180px;
    display: flex; flex-direction: column; gap: 22px;
  }
  .msg{ display: flex; flex-direction: column; animation: fadeUp .28s ease }
  @keyframes fadeUp{ from{ opacity: 0; transform: translateY(4px) } to{ opacity: 1; transform: none } }

  .msg .bubble-wrap{ display: flex; align-items: flex-start; gap: 10px }
  .msg.user .bubble-wrap{ justify-content: flex-end }
  .avatar{
    width: 28px; height: 28px; border-radius: 50%;
    background: var(--surface-3);
    color: var(--text-muted);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; letter-spacing: .05em;
    flex-shrink: 0; margin-top: 2px;
    border: 1px solid var(--border);
  }
  .msg.advisor .avatar{ background: var(--accent-bg); color: var(--accent); border-color: var(--border-strong) }

  .bubble-content{ min-width: 0; max-width: 100% }
  .msg-meta{
    font-size: 11px; color: var(--text-subtle);
    letter-spacing: .05em; margin-bottom: 6px;
    display: flex; align-items: center; gap: 8px;
  }
  .msg-meta .branch-mark{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 1px 8px; border-radius: 999px;
    background: var(--surface-3); color: var(--text-muted);
    font-size: 10px;
  }
  .msg-meta .branch-mark svg{ width: 10px; height: 10px }

  .bubble{
    padding: 14px 16px;
    border-radius: var(--radius-lg);
    line-height: 1.65;
    overflow-wrap: break-word;
  }
  .msg.user .bubble{
    background: var(--user-bubble);
    border: 1px solid var(--border);
    max-width: 720px;
    white-space: pre-wrap;
  }
  .msg.advisor .bubble{
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
  }
  .msg.advisor .bubble.thinking{
    padding: 12px 16px;
    display: inline-flex; align-items: center; gap: 10px;
    color: var(--text-muted); font-style: italic;
  }
  .dots{ display: inline-flex; gap: 3px }
  .dots span{
    width: 5px; height: 5px; border-radius: 50%;
    background: currentColor; opacity: .4;
    animation: dot 1.2s infinite ease-in-out;
  }
  .dots span:nth-child(2){ animation-delay: .15s }
  .dots span:nth-child(3){ animation-delay: .3s }
  @keyframes dot{ 0%,80%,100%{ opacity: .2 } 40%{ opacity: .95 } }

  /* Message actions */
  .bubble-actions{
    display: flex; gap: 2px; margin-top: 6px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .msg:hover .bubble-actions{ opacity: 1 }
  .bubble-actions .icon-btn{ width: 28px; height: 28px; color: var(--text-subtle) }
  .bubble-actions .icon-btn:hover{ color: var(--accent) }
  .bubble-actions .icon-btn svg{ width: 13px; height: 13px }

  /* Typography inside bubble */
  .bubble h1{ font-family: var(--font-brand); font-weight: 500; font-size: 22px; letter-spacing: .05em; margin: 4px 0 12px }
  .bubble h2{ font-family: var(--font-brand); font-weight: 500; font-size: 18px; letter-spacing: .04em; margin: 18px 0 8px }
  .bubble h3{ font-weight: 600; font-size: 15.5px; margin: 14px 0 6px }
  .bubble h4,.bubble h5,.bubble h6{ font-weight: 600; font-size: 14px; margin: 12px 0 4px }
  .bubble p{ margin: 8px 0 }
  .bubble p:first-child{ margin-top: 0 }
  .bubble p:last-child{ margin-bottom: 0 }
  .bubble ul,.bubble ol{ margin: 8px 0 8px 22px }
  .bubble li{ margin: 3px 0 }
  .bubble strong{ font-weight: 600 }
  .bubble code{
    font-family: var(--font-mono); font-size: .9em;
    background: var(--surface-2); border: 1px solid var(--border);
    padding: 1px 6px; border-radius: 4px;
  }
  .bubble pre{
    background: var(--surface-2); border: 1px solid var(--border);
    padding: 12px 14px; border-radius: var(--radius);
    font-family: var(--font-mono); font-size: 13px;
    overflow: auto; margin: 10px 0;
  }
  .bubble blockquote{
    border-left: 3px solid var(--accent);
    padding: 4px 14px; margin: 10px 0;
    background: var(--surface-2);
    color: var(--text-muted);
  }
  .bubble table{ width: 100%; border-collapse: collapse; margin: 10px 0 }
  .bubble th,.bubble td{ border: 1px solid var(--border); padding: 8px 10px; text-align: left; font-size: 13.5px }
  .bubble th{ background: var(--surface-2); font-weight: 600 }

  /* Welcome card */
  .welcome{
    text-align: center;
    padding: 72px 20px 20px;
    color: var(--text-muted);
  }
  .welcome-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 28px; letter-spacing: .08em;
    color: var(--text);
    margin-bottom: 8px;
  }
  .welcome-sub{ font-size: 14px; max-width: 540px; margin: 0 auto; line-height: 1.7 }
  .suggestions{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 10px; max-width: 720px; margin: 28px auto 0;
  }
  .suggestion{
    padding: 14px 16px; border: 1px solid var(--border);
    border-radius: var(--radius); background: var(--surface);
    text-align: left; cursor: pointer; font-size: 13.5px;
    transition: border-color var(--t-fast), background var(--t-fast), transform var(--t-fast);
    color: var(--text);
  }
  .suggestion:hover{ border-color: var(--accent); background: var(--accent-bg); transform: translateY(-1px) }
  .suggestion .title{ font-weight: 600; margin-bottom: 4px }
  .suggestion .desc{ color: var(--text-muted); font-size: 12.5px }

  /* ============ Composer ============ */
  .composer-wrap{
    position: fixed; bottom: 0;
    left: var(--rail-w); right: var(--tree-w);
    padding: 18px 20px 22px;
    background: linear-gradient(to top, var(--bg) 60%, transparent);
    transition: left var(--t-base), right var(--t-base);
    pointer-events: none;
  }
  body.sidebar-closed .composer-wrap{ left: 0 }
  body.tree-closed    .composer-wrap{ right: 0 }

  .composer{
    max-width: 860px; margin: 0 auto;
    pointer-events: auto;
  }
  .branch-indicator{
    display: none;
    align-items: center; gap: 8px;
    padding: 6px 12px; margin-bottom: 8px;
    background: var(--accent-bg); border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    font-size: 12px; color: var(--text-muted);
  }
  .branch-indicator.show{ display: inline-flex }
  .branch-indicator svg{ width: 12px; height: 12px; color: var(--accent) }
  .branch-indicator button{
    margin-left: auto; color: var(--text-muted); font-size: 11px;
    text-decoration: underline;
  }

  .composer-bar{
    display: flex; align-items: flex-end; gap: 8px;
    padding: 10px 10px 10px 16px;
    background: var(--surface);
    border: 1px solid var(--border-strong);
    border-radius: 20px;
    box-shadow: var(--shadow);
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .composer-bar:focus-within{ border-color: var(--accent); box-shadow: 0 0 0 3px var(--ring), var(--shadow) }

  .composer-input{
    flex: 1;
    min-height: 28px; max-height: 200px;
    outline: 0;
    font: 15px/1.55 var(--font-sans); color: var(--text);
    padding: 6px 0;
    overflow-y: auto;
    overflow-wrap: break-word;
    word-break: break-word;
  }
  .composer-input:empty::before{
    content: attr(data-placeholder);
    color: var(--text-subtle);
  }

  .composer-actions{ display: flex; align-items: center; gap: 4px }
  .composer-actions .icon-btn{ width: 36px; height: 36px; border-radius: 10px }
  .send-btn{
    width: 38px; height: 38px; border-radius: 10px;
    background: var(--text); color: var(--bg);
    display: inline-flex; align-items: center; justify-content: center;
    transition: background var(--t-fast), transform var(--t-fast);
  }
  .send-btn:hover{ background: var(--accent); color: #fff }
  .send-btn:disabled{ background: var(--surface-3); color: var(--text-subtle); cursor: not-allowed }
  .send-btn svg{ width: 16px; height: 16px }

  .composer-foot{
    max-width: 860px; margin: 8px auto 0;
    font-size: 11.5px; color: var(--text-subtle);
    display: flex; align-items: center; justify-content: space-between; gap: 10px;
  }
  .file-chips{ display: flex; flex-wrap: wrap; gap: 6px }
  .chip{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 8px; border: 1px solid var(--border);
    border-radius: 999px; background: var(--surface);
    font-size: 11px;
  }
  .chip svg{ width: 11px; height: 11px }
  .chip button{ color: var(--text-subtle); line-height: 1 }
  .chip button:hover{ color: var(--error) }

  #file{ display: none }

  /* ============ Tree pane ============ */
  .tree-head{
    height: 56px; padding: 0 16px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .tree-title{
    font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-muted); font-weight: 600;
  }
  .tree-body{ flex: 1; overflow: auto; padding: 14px 12px 24px }

  .tree-empty{ padding: 28px 16px; text-align: center; color: var(--text-subtle); font-size: 13px }

  /* Tree nodes — indent + connector */
  .tree-node{
    position: relative;
    padding: 6px 8px 6px 22px;
    margin: 2px 0;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--t-fast);
  }
  .tree-node:hover{ background: var(--surface-2) }
  .tree-node.active{ background: var(--accent-bg) }
  .tree-node.in-path{ background: linear-gradient(to right, var(--surface-2), transparent 60%) }

  /* vertical connector line */
  .tree-node::before{
    content: ""; position: absolute;
    left: 10px; top: 0; bottom: 0;
    width: 1px; background: var(--border);
  }
  /* horizontal tick */
  .tree-node::after{
    content: ""; position: absolute;
    left: 10px; top: 18px; width: 10px; height: 1px;
    background: var(--border);
  }
  .tree-node[data-depth="0"]::before,
  .tree-node[data-depth="0"]::after{ display: none }
  .tree-node[data-depth="0"]{ padding-left: 8px }

  .tree-node .dot{
    position: absolute; left: 6px; top: 13px;
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--surface); border: 1.5px solid var(--border-strong);
    z-index: 1;
  }
  .tree-node[data-depth="0"] .dot{ left: auto; position: static; margin-right: 8px; display: inline-block; vertical-align: middle }
  .tree-node.active .dot,
  .tree-node.in-path .dot{ background: var(--accent); border-color: var(--accent) }
  .tree-node.assistant .dot{ background: var(--surface); border-color: var(--accent); border-style: dashed }
  .tree-node.assistant.active .dot{ background: var(--accent) }

  .tree-row{
    display: flex; align-items: center; gap: 8px;
    min-width: 0;
  }
  .tree-label{
    flex: 1; font-size: 13px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    color: var(--text);
  }
  .tree-node.assistant .tree-label{ color: var(--text-muted); font-size: 12.5px }
  .tree-node .role-tag{
    font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase;
    color: var(--text-subtle);
    margin-right: 6px;
  }
  .tree-actions{
    display: flex; gap: 2px;
    opacity: 0; transition: opacity var(--t-fast);
  }
  .tree-node:hover .tree-actions,
  .tree-node.active .tree-actions{ opacity: 1 }
  .tree-actions .icon-btn{ width: 24px; height: 24px }
  .tree-actions .icon-btn svg{ width: 12px; height: 12px }

  /* ============ Modal ============ */
  .modal-root{
    position: fixed; inset: 0; z-index: 60;
    display: none; align-items: center; justify-content: center;
    background: rgba(15,15,15,.34);
    backdrop-filter: blur(4px);
    animation: fadeIn .18s ease;
  }
  .modal-root.show{ display: flex }
  .modal{
    width: min(420px, 92vw);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: 22px 22px 18px;
  }
  .modal h3{ font-family: var(--font-brand); font-weight: 500; font-size: 17px; letter-spacing: .05em; margin-bottom: 6px }
  .modal p{ color: var(--text-muted); font-size: 13.5px; margin-bottom: 14px }
  .modal .field{ margin-bottom: 14px }
  .modal-actions{ display: flex; justify-content: flex-end; gap: 8px }
  @keyframes fadeIn{ from{ opacity: 0 } to{ opacity: 1 } }

  /* ============ Graph view ============ */
  .graph-root{
    position: fixed; inset: 0;
    display: none; flex-direction: column;
    background: var(--bg);
    z-index: 80;
    animation: fadeIn .18s ease;
  }
  .graph-root.show{ display: flex }
  .graph-head{
    height: 64px; padding: 0 20px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  .graph-title{
    font-family: var(--font-brand);
    font-weight: 500; font-size: 17px; letter-spacing: .08em;
  }
  .graph-sub{ color: var(--text-subtle); font-size: 11.5px; letter-spacing: .02em; margin-top: 2px }
  .graph-actions{ display: flex; align-items: center; gap: 4px }
  .graph-sep{ width: 1px; height: 22px; background: var(--border); margin: 0 6px }
  .graph-stage{
    flex: 1; min-height: 0; position: relative;
    overflow: hidden;
    background:
      radial-gradient(circle at 1px 1px, var(--border) 1px, transparent 0) 0 0/24px 24px,
      var(--bg);
    cursor: grab;
  }
  .graph-stage.dragging{ cursor: grabbing }
  .graph-svg{ width: 100%; height: 100%; display: block; user-select: none; }
  .graph-empty{
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    color: var(--text-subtle); font-size: 14px;
    pointer-events: none;
  }
  .graph-empty.hide{ display: none }

  /* graph edges */
  .graph-edge{ fill: none; stroke: var(--border); stroke-width: 1.5 }
  .graph-edge.in-path{ stroke: var(--accent); stroke-width: 2 }

  /* graph nodes */
  .graph-node{ cursor: pointer; transition: transform .15s ease }
  .graph-node:hover{ transform: translateZ(0) }
  .graph-node .node-bg{
    fill: var(--surface);
    stroke: var(--border);
    stroke-width: 1.5;
    transition: stroke .15s, stroke-width .15s, filter .15s;
  }
  .graph-node:hover .node-bg{ stroke: var(--accent); filter: drop-shadow(0 3px 10px rgba(0,0,0,.15)) }
  .graph-node.user .node-bg{ fill: var(--surface-2) }
  .graph-node.advisor .node-bg{ fill: var(--accent-bg) }
  .graph-node.in-path .node-bg{ stroke: var(--accent); stroke-width: 2 }
  .graph-node.active .node-bg{ stroke: var(--accent); stroke-width: 2.5; fill: var(--accent-bg) }

  .graph-node .node-dot{ fill: var(--text-muted) }
  .graph-node.user .node-dot{ fill: var(--text-muted) }
  .graph-node.advisor .node-dot{ fill: var(--accent) }
  .graph-node.in-path .node-dot,
  .graph-node.active .node-dot{ fill: var(--accent) }

  .graph-node .node-label{
    fill: var(--text); font-size: 11.5px;
    font-family: var(--font-sans);
  }
  .graph-node.user .node-label{ fill: var(--text) }
  .graph-node.advisor .node-label{ fill: var(--text-muted) }

  .graph-node .node-role{
    fill: var(--text-subtle); font-size: 9.5px;
    letter-spacing: .14em; text-transform: uppercase;
    font-weight: 600;
  }

  /* branch labels (above subtree roots with siblings) */
  .graph-branch-label{
    fill: var(--accent); font-size: 10.5px; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
  }

  .graph-legend{
    padding: 10px 20px; border-top: 1px solid var(--border);
    background: var(--surface);
    display: flex; align-items: center; gap: 18px;
    font-size: 12px; color: var(--text-muted);
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  .graph-legend .dot{
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
    border: 1px solid var(--border);
  }
  .graph-legend .dot.user{ background: var(--surface-2) }
  .graph-legend .dot.advisor{ background: var(--accent-bg); border-color: var(--border-strong) }
  .graph-legend .dot.active{ background: var(--accent); border-color: var(--accent) }
  .graph-legend .dot.topic{ background: transparent; border: 1.5px dashed var(--accent) }
  .graph-legend .graph-hint{ margin-left: auto; color: var(--text-subtle); font-size: 11px }

  /* ---- Toolbar ---- */
  .graph-toolbar{
    display: inline-flex; align-items: center; gap: 2px;
    padding: 3px; border-radius: var(--radius);
    background: var(--surface-2); border: 1px solid var(--border);
  }
  .tool-btn{
    width: 34px; height: 30px; border-radius: 7px;
    display: inline-flex; align-items: center; justify-content: center;
    color: var(--text-muted); transition: background var(--t-fast), color var(--t-fast);
  }
  .tool-btn svg{ width: 15px; height: 15px }
  .tool-btn:hover{ background: var(--surface-3); color: var(--text) }
  .tool-btn.active{ background: var(--accent); color: #fff }
  .tool-btn.active:hover{ background: var(--accent) }

  /* ---- Stage cursors per mode ---- */
  .graph-stage.mode-select{ cursor: grab }
  .graph-stage.mode-select.dragging{ cursor: grabbing }
  .graph-stage.mode-zone{ cursor: crosshair }
  .graph-stage.mode-topic{ cursor: copy }

  /* ---- Zones ---- */
  .graph-zone{ cursor: pointer; transition: opacity var(--t-fast) }
  .graph-zone:hover rect{ fill-opacity: 0.18 }
  .graph-zone.selected rect{ filter: drop-shadow(0 4px 14px rgba(0,0,0,.15)) }
  .zone-title{
    font-size: 10.5px; font-weight: 700;
    letter-spacing: .14em; font-family: var(--font-sans);
    pointer-events: none;
  }
  .zone-resize{ cursor: nwse-resize; fill-opacity: 0.85; rx: 2; stroke: #fff; stroke-width: 1 }
  .ghost-zone{
    fill: var(--accent); fill-opacity: 0.10;
    stroke: var(--accent); stroke-width: 1.5;
    stroke-dasharray: 6 4;
    pointer-events: none;
  }

  /* ---- Topic nodes ---- */
  .graph-node.topic .node-bg{
    fill: var(--surface);
    stroke: var(--accent);
    stroke-width: 1.5;
    stroke-dasharray: 5 3;
  }
  .graph-node.topic .node-dot{ fill: var(--accent) }
  .graph-node.topic .node-label{ font-weight: 600 }

  /* ---- Selected / pinned / dragging nodes ---- */
  .graph-node.selected .node-bg{
    stroke: var(--accent);
    stroke-width: 2.5;
    filter: drop-shadow(0 6px 18px rgba(139,111,62,.35));
  }
  .graph-node.dragging{ opacity: 0.85 }
  .graph-node.dragging .node-bg{ filter: drop-shadow(0 10px 22px rgba(0,0,0,.25)) }
  .node-pin{ fill: var(--accent); opacity: 0.85 }
  .node-narrative-badge{ fill: var(--accent); opacity: 0.85 }

  /* ---- Narrative side panel ---- */
  .narrative-panel{
    position: absolute; top: 0; right: 0; bottom: 0;
    width: 420px; max-width: 90vw;
    background: var(--surface);
    border-left: 1px solid var(--border);
    box-shadow: var(--shadow-lg);
    display: flex; flex-direction: column;
    transform: translateX(100%);
    transition: transform var(--t-base);
    z-index: 5;
  }
  .narrative-panel.show{ transform: translateX(0) }
  .narr-head{
    height: 52px; padding: 0 14px 0 18px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .narr-kind{
    font-size: 10.5px; font-weight: 700;
    letter-spacing: .16em; color: var(--accent);
  }
  .narr-body{
    flex: 1; min-height: 0; overflow-y: auto;
    padding: 18px 20px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .narr-label{
    font-size: 10.5px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--text-subtle); font-weight: 600;
    margin-top: 10px;
  }
  .narr-label:first-child{ margin-top: 0 }
  .narr-input, .narr-textarea{
    width: 100%;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    color: var(--text); font-size: 13.5px;
    font-family: var(--font-sans);
    transition: border-color var(--t-fast), box-shadow var(--t-fast);
  }
  .narr-input:focus, .narr-textarea:focus{
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--ring);
  }
  .narr-textarea{ resize: vertical; min-height: 160px; line-height: 1.55 }
  .color-row{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 2px }
  .color-sw{
    width: 26px; height: 26px; border-radius: 50%;
    border: 2px solid transparent;
    cursor: pointer;
    transition: transform var(--t-fast), border-color var(--t-fast);
  }
  .color-sw:hover{ transform: scale(1.1) }
  .color-sw.active{ border-color: var(--text); box-shadow: 0 0 0 1px var(--surface) inset }
  .narr-foot{
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px;
    flex-shrink: 0;
  }

  @media (max-width: 640px){
    .graph-legend{ gap: 10px; font-size: 11px }
    .graph-legend .graph-hint{ width: 100%; margin-left: 0; margin-top: 4px }
    .narrative-panel{ width: 100% }
    .graph-toolbar{ padding: 2px }
    .tool-btn{ width: 30px; height: 28px }
  }

  /* ============ Toasts ============ */
  .toast-root{
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; gap: 8px; z-index: 70;
    pointer-events: none;
  }
  .toast{
    background: var(--surface); border: 1px solid var(--border);
    box-shadow: var(--shadow-lg);
    border-radius: var(--radius);
    padding: 10px 14px; font-size: 13px;
    display: flex; align-items: center; gap: 10px;
    animation: toastIn .22s ease;
    pointer-events: auto;
  }
  .toast.error{ border-color: var(--error); color: var(--error) }
  .toast.success{ border-color: var(--success); color: var(--success) }
  .toast svg{ width: 14px; height: 14px }
  @keyframes toastIn{ from{ opacity: 0; transform: translateY(6px) } to{ opacity: 1; transform: none } }

  /* ============ Scrollbars ============ */
  ::-webkit-scrollbar{ width: 10px; height: 10px }
  ::-webkit-scrollbar-track{ background: transparent }
  ::-webkit-scrollbar-thumb{ background: var(--border-strong); border: 2px solid transparent; background-clip: padding-box; border-radius: 10px }
  ::-webkit-scrollbar-thumb:hover{ background: var(--text-subtle); background-clip: padding-box; border: 2px solid transparent }

  /* ============ Mobile ============ */
  @media (max-width: 980px){
    :root{ --rail-w: 0px; --tree-w: 0px }
    .edge-tab{ display: none }
    .composer-wrap{ left: 0; right: 0 }
  }
</style>
</head>
<body>
  <div class="app">
    <!-- ============ LEFT RAIL ============ -->
    <aside class="rail" id="rail">
      <div class="brand">
        <div>
          <div class="brand-title">Fiduciary Advisor</div>
          <div class="brand-sub">Private Trust Counsel</div>
        </div>
        <button class="icon-btn" id="btn-rail-collapse" title="Collapse sidebar" aria-label="Collapse sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
        </button>
      </div>

      <div class="rail-section">
        <div class="field-label">Client Profile</div>
        <div class="row">
          <div class="field">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-subtle)"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
            <input id="pf-id" placeholder="Client ID (required)"/>
          </div>
        </div>
        <div class="row">
          <div class="field">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;color:var(--text-subtle)"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><path d="M22 6l-10 7L2 6"/></svg>
            <input id="pf-email" placeholder="Email (optional)"/>
          </div>
        </div>
        <div class="row" style="margin-top:10px">
          <button class="btn primary full" id="pf-save">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
            Save
          </button>
          <button class="btn ghost" id="pf-logout" title="Log out">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
          </button>
        </div>
        <div class="msg-line" id="pf-msg"></div>
      </div>

      <div class="rail-section" style="padding-top:10px; padding-bottom:10px">
        <button class="btn accent full" id="btn-newchat">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
          New Chat
        </button>
      </div>

      <div class="chat-list-header" style="padding: 10px 16px 0">
        <div class="field-label" style="margin:0">Conversations</div>
      </div>
      <div class="chat-list" id="chatlist"></div>

      <div class="rail-section" style="display:flex; justify-content:space-between; align-items:center">
        <div style="font-size: 11px; color: var(--text-subtle); letter-spacing: .06em">api.fctadvisor.com</div>
        <button class="icon-btn" id="btn-theme" title="Toggle theme">
          <svg id="ic-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
        </button>
      </div>
    </aside>

    <!-- ============ MAIN ============ -->
    <main class="main">
      <div class="main-header">
        <button class="icon-btn" id="btn-toggle-rail-m" title="Toggle sidebar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
        </button>
        <div class="main-title" id="chat-title">
          <span class="muted">Advisor</span>
        </div>
        <button class="icon-btn" id="btn-toggle-tree-m" title="Toggle tree">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="6" cy="18" r="2"/><path d="M8 6h4a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8"/><path d="M16 12H8"/></svg>
        </button>
      </div>

      <div class="thread-scroll" id="thread-scroll">
        <div class="thread" id="thread"></div>
      </div>
    </main>

    <!-- ============ TREE PANE ============ -->
    <aside class="tree-pane" id="treePane">
      <div class="tree-head">
        <div class="tree-title">Conversation Tree</div>
        <div style="display:flex; align-items:center; gap:4px">
          <button class="icon-btn" id="btn-open-graph" title="Open graph view">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="19" cy="12" r="2"/><path d="M7 6h6a3 3 0 0 1 3 3v0a3 3 0 0 1-3 3H10"/><path d="M7 18h6a3 3 0 0 0 3-3"/></svg>
          </button>
          <button class="icon-btn" id="btn-tree-collapse" title="Collapse tree">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
          </button>
        </div>
      </div>
      <div class="tree-body" id="tree-body">
        <div class="tree-empty">Send a message to begin a thread.</div>
      </div>
    </aside>

    <!-- Edge tabs -->
    <button class="edge-tab left"  id="left-tab"  title="Open sidebar" aria-label="Open sidebar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18l6-6-6-6"/></svg>
    </button>
    <button class="edge-tab right" id="right-tab" title="Open tree" aria-label="Open tree">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
    </button>
  </div>

  <!-- Composer -->
  <div class="composer-wrap">
    <div class="composer">
      <div class="branch-indicator" id="branch-ind">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
        <span id="branch-ind-text">Branching from a previous message…</span>
        <button id="branch-clear">Cancel</button>
      </div>
      <div class="composer-bar">
        <div class="composer-input" id="input" contenteditable="true" data-placeholder="Pose an inquiry to the Advisor… (Shift+Enter for new line)"></div>
        <div class="composer-actions">
          <button class="icon-btn" id="attach" title="Attach files">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
          </button>
          <input id="file" type="file" multiple accept=".pdf,.txt,.docx"/>
          <button class="send-btn" id="send" title="Send" aria-label="Send">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
      </div>
      <div class="composer-foot">
        <div id="foot-hint">Formal trust, fiduciary, and contractual analysis with strategic guidance.</div>
        <div class="file-chips" id="file-chips"></div>
      </div>
    </div>
  </div>

  <!-- Modal root -->
  <div class="modal-root" id="modal-root">
    <div class="modal" id="modal">
      <h3 id="modal-title">Rename</h3>
      <p id="modal-desc"></p>
      <div class="field">
        <input id="modal-input" placeholder=""/>
      </div>
      <div class="modal-actions">
        <button class="btn ghost" id="modal-cancel">Cancel</button>
        <button class="btn primary" id="modal-ok">Confirm</button>
      </div>
    </div>
  </div>

  <!-- Graph overlay -->
  <div class="graph-root" id="graph-root">
    <div class="graph-head">
      <div>
        <div class="graph-title">Conversation Graph</div>
        <div class="graph-sub" id="graph-sub">Click to select · Drag to reposition · Double-click empty canvas for new topic · Shift-click a node to branch</div>
      </div>
      <div class="graph-actions">
        <!-- Tool selector -->
        <div class="graph-toolbar">
          <button class="tool-btn active" data-graph-tool="select" title="Select & move">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l7.5 18 2-7.5 7.5-2z"/></svg>
          </button>
          <button class="tool-btn" data-graph-tool="zone" title="Draw zone (click-drag on empty canvas)">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3" stroke-dasharray="4 3"/></svg>
          </button>
        </div>
        <span class="graph-sep"></span>
        <button class="icon-btn" id="graph-relayout" title="Reset to auto-layout">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 15.5-6.2"/><path d="M21 4v6h-6"/><path d="M21 12a9 9 0 0 1-15.5 6.2"/><path d="M3 20v-6h6"/></svg>
        </button>
        <span class="graph-sep"></span>
        <button class="icon-btn" id="graph-zoom-out" title="Zoom out">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M8 11h6"/></svg>
        </button>
        <button class="icon-btn" id="graph-zoom-fit" title="Fit view">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V5a1 1 0 0 1 1-1h4"/><path d="M20 9V5a1 1 0 0 0-1-1h-4"/><path d="M4 15v4a1 1 0 0 0 1 1h4"/><path d="M20 15v4a1 1 0 0 1-1 1h-4"/></svg>
        </button>
        <button class="icon-btn" id="graph-zoom-in" title="Zoom in">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M11 8v6"/><path d="M8 11h6"/></svg>
        </button>
        <span class="graph-sep"></span>
        <button class="icon-btn" id="graph-close" title="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18"/><path d="M6 6l12 12"/></svg>
        </button>
      </div>
    </div>
    <div class="graph-stage mode-select" id="graph-stage">
      <svg class="graph-svg" id="graph-svg" xmlns="http://www.w3.org/2000/svg"></svg>
      <div class="graph-empty" id="graph-empty">No messages yet. Start a conversation, or double-click the canvas to add a topic.</div>

      <!-- Narrative side panel -->
      <aside class="narrative-panel" id="narrative-panel">
        <div class="narr-head">
          <span class="narr-kind" id="narr-kind">NODE</span>
          <button class="icon-btn" id="narr-close" title="Close panel">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18"/><path d="M6 6l12 12"/></svg>
          </button>
        </div>
        <div class="narr-body">
          <label class="narr-label">Title</label>
          <input class="narr-input" id="narr-title" placeholder="Name this node or zone"/>

          <div id="narr-color-row-wrap" style="display:none">
            <label class="narr-label">Color</label>
            <div class="color-row" id="narr-color-row"></div>
          </div>

          <label class="narr-label">Narrative</label>
          <textarea class="narr-textarea" id="narr-text" rows="10" placeholder="Comprehensive notes, context, reasoning, decisions, references…"></textarea>
        </div>
        <div class="narr-foot">
          <button class="btn ghost" id="narr-delete">Delete</button>
          <div style="flex:1"></div>
          <button class="btn ghost" id="narr-open-chat">Open in chat</button>
          <button class="btn primary" id="narr-save">Save</button>
        </div>
      </aside>
    </div>
    <div class="graph-legend">
      <span><i class="dot user"></i> Your question</span>
      <span><i class="dot advisor"></i> Advisor reply</span>
      <span><i class="dot topic"></i> Topic</span>
      <span><i class="dot active"></i> Active path</span>
      <span class="graph-hint">Drag background = pan · Drag node = reposition · Double-click = new topic · Scroll = zoom</span>
    </div>
  </div>

  <!-- Toast root -->
  <div class="toast-root" id="toast-root"></div>

<script>
/* ===================== State ===================== */
const LS = {
  UI:   'fct.ui.v2',
  USER: 'fct.userId',
  EMAIL:'fct.email',
  THEME:'fct.theme'
};
const ui = (() => {
  const def = { sidebarOpen: true, treeOpen: true };
  try { return Object.assign(def, JSON.parse(localStorage.getItem(LS.UI) || '{}')); }
  catch(_){ return def; }
})();
const saveUI = () => localStorage.setItem(LS.UI, JSON.stringify(ui));

const state = {
  userId:  localStorage.getItem(LS.USER) || '',
  email:   localStorage.getItem(LS.EMAIL) || '',
  theme:   localStorage.getItem(LS.THEME) || 'light',
  currentChatId: null,
  currentChatTitle: '',
  treeNodes: [],         // raw message list for current chat
  activeLeafId: null,    // the leaf representing the currently-viewed path
  branchFromId: null,    // if set, next send will fork under this msg
};

/* ===================== Theme ===================== */
function applyTheme(){
  document.documentElement.setAttribute('data-theme', state.theme);
  const ic = document.getElementById('ic-sun');
  if (state.theme === 'dark'){
    ic.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
  } else {
    ic.innerHTML = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>';
  }
}

/* ===================== Layout ===================== */
function applyLayout(){
  document.body.classList.toggle('sidebar-closed', !ui.sidebarOpen);
  document.body.classList.toggle('tree-closed', !ui.treeOpen);
}

/* ===================== Elements ===================== */
const elThread    = document.getElementById('thread');
const elThreadS   = document.getElementById('thread-scroll');
const elChatTitle = document.getElementById('chat-title');
const elInput     = document.getElementById('input');
const elSend      = document.getElementById('send');
const elAttach    = document.getElementById('attach');
const elFile      = document.getElementById('file');
const elChips     = document.getElementById('file-chips');
const elPfId      = document.getElementById('pf-id');
const elPfEmail   = document.getElementById('pf-email');
const elPfMsg     = document.getElementById('pf-msg');
const elChatList  = document.getElementById('chatlist');
const elTreeBody  = document.getElementById('tree-body');
const elBranchInd = document.getElementById('branch-ind');
const elBranchText= document.getElementById('branch-ind-text');

elPfId.value = state.userId || '';
elPfEmail.value = state.email || '';

/* ===================== Toasts ===================== */
function toast(msg, kind='info'){
  const root = document.getElementById('toast-root');
  const el = document.createElement('div');
  el.className = 'toast ' + (kind || '');
  const icon = kind === 'error'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    : kind === 'success'
    ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
    : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>';
  el.innerHTML = icon + '<span>' + msg + '</span>';
  root.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transform = 'translateY(6px)'; }, 2400);
  setTimeout(() => el.remove(), 2700);
}

/* ===================== Modal ===================== */
function modal({ title, desc, placeholder = '', value = '', okLabel = 'Confirm' } = {}){
  return new Promise(resolve => {
    const root = document.getElementById('modal-root');
    document.getElementById('modal-title').textContent = title || 'Confirm';
    document.getElementById('modal-desc').textContent  = desc || '';
    const input = document.getElementById('modal-input');
    input.placeholder = placeholder;
    input.value = value;
    document.getElementById('modal-ok').textContent = okLabel;
    root.classList.add('show');
    setTimeout(() => input.focus(), 50);

    function close(val){
      root.classList.remove('show');
      document.getElementById('modal-ok').onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      input.onkeydown = null;
      resolve(val);
    }
    document.getElementById('modal-ok').onclick = () => close(input.value.trim());
    document.getElementById('modal-cancel').onclick = () => close(null);
    input.onkeydown = e => {
      if (e.key === 'Enter') close(input.value.trim());
      if (e.key === 'Escape') close(null);
    };
  });
}
function confirmDialog({ title, desc, okLabel = 'Confirm' } = {}){
  return new Promise(resolve => {
    const root = document.getElementById('modal-root');
    document.getElementById('modal-title').textContent = title || 'Confirm';
    document.getElementById('modal-desc').textContent  = desc || '';
    const input = document.getElementById('modal-input');
    input.style.display = 'none'; input.parentElement.style.display = 'none';
    const ok = document.getElementById('modal-ok');
    ok.textContent = okLabel;
    root.classList.add('show');

    function restore(){ input.style.display = ''; input.parentElement.style.display = '' }
    function close(val){
      root.classList.remove('show'); restore();
      ok.onclick = null;
      document.getElementById('modal-cancel').onclick = null;
      resolve(val);
    }
    ok.onclick = () => close(true);
    document.getElementById('modal-cancel').onclick = () => close(false);
  });
}

/* ===================== Fetch helpers ===================== */
function hdrs(){
  const h = {};
  if (state.userId) h['X-User-Id'] = state.userId;
  return h;
}
async function api(method, url, { body, form } = {}){
  const opts = { method, headers: { ...hdrs() } };
  if (form){ opts.body = form; }
  else if (body){
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(url, opts);
  if (!r.ok){
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch(_){}
    throw new Error(msg + ' (' + r.status + ')');
  }
  return r.json();
}

/* ===================== Formatting ===================== */
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])) }

function mdToHtml(md){
  if (!md) return '';
  if (/<\w+[^>]*>/.test(md)) return md;
  let h = md.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  h = h.replace(/```([\s\S]*?)```/g, (_, c) => '<pre><code>' + c + '</code></pre>');
  h = h.replace(/`([^`]+?)`/g, '<code>$1</code>');
  h = h.replace(/^######\s+(.*)$/gm, '<h6>$1</h6>')
       .replace(/^#####\s+(.*)$/gm,  '<h5>$1</h5>')
       .replace(/^####\s+(.*)$/gm,   '<h4>$1</h4>')
       .replace(/^###\s+(.*)$/gm,    '<h3>$1</h3>')
       .replace(/^##\s+(.*)$/gm,     '<h2>$1</h2>')
       .replace(/^#\s+(.*)$/gm,      '<h1>$1</h1>');
  h = h.replace(/^>\s?(.*)$/gm, '<blockquote>$1</blockquote>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
       .replace(/__(.+?)__/g,     '<strong>$1</strong>')
       .replace(/\*(?!\s)(.+?)\*/g, '<em>$1</em>')
       .replace(/_(?!\s)(.+?)_/g,  '<em>$1</em>');
  h = h.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  h = h.replace(/\n{2,}/g, '</p><p>').replace(/^(?!<h\d|<ul|<ol|<pre|<hr|<p|<blockquote|<table)(.+)$/gm, '<p>$1</p>');
  return h;
}

function normalizeTrustDoc(html){
  let out = html;
  out = out.replace(/<p>\s*<strong>\s*([A-Z0-9][A-Z0-9\s\-&,.'()]+?)\s*<\/strong>\s*<\/p>/g, '<h2>$1</h2>');
  const labelMap = [
    [/\*\*\s*TRUST\s*NAME\s*:\s*\*\*/gi, 'Trust: '],
    [/\*\*\s*DATE\s*:\s*\*\*/gi,         'Date: '],
    [/\*\*\s*TAX\s*YEAR\s*:\s*\*\*/gi,   'Tax Year: '],
    [/\*\*\s*TRUSTEE\(S\)\s*:\s*\*\*/gi, 'Trustee(s): '],
    [/\*\*\s*LOCATION\s*:\s*\*\*/gi,     'Location: '],
  ];
  labelMap.forEach(([re, rep]) => { out = out.replace(re, rep) });
  return out;
}

/* ===================== Thread rendering ===================== */
function now(){ return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }

function pathFromLeaf(leafId){
  // walk up parent_ids from leaf to root, collecting messages
  if (!leafId) return [];
  const byId = {};
  state.treeNodes.forEach(n => byId[n.id] = n);
  const out = [];
  let cur = byId[leafId];
  const guard = new Set();
  while (cur && !guard.has(cur.id)){
    guard.add(cur.id);
    out.unshift(cur);
    cur = cur.parent_id ? byId[cur.parent_id] : null;
  }
  return out;
}

function leaves(){
  // ids that no other node claims as parent
  const hasChild = new Set();
  state.treeNodes.forEach(n => { if (n.parent_id) hasChild.add(n.parent_id) });
  return state.treeNodes.filter(n => !hasChild.has(n.id));
}

function plainText(html){
  const d = document.createElement('div');
  d.innerHTML = html || '';
  return (d.textContent || '').replace(/\s+/g, ' ').trim();
}

function renderThread(){
  elThread.innerHTML = '';
  const path = pathFromLeaf(state.activeLeafId);
  if (!path.length){
    elThread.innerHTML = welcomeHTML();
    bindSuggestions();
    return;
  }
  path.forEach(n => {
    const isUser = n.role === 'user';
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + (isUser ? 'user' : 'advisor');
    wrap.dataset.id = n.id;

    const avatar = isUser
      ? '<div class="avatar">You</div>'
      : '<div class="avatar">FA</div>';

    const siblings = state.treeNodes.filter(m => m.parent_id === n.parent_id);
    const hasBranches = siblings.length > 1;
    const branchMark = hasBranches
      ? `<span class="branch-mark" title="${siblings.length} alternatives">
           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
           ${siblings.length}
         </span>`
      : '';

    const time = new Date(n.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const meta = `<div class="msg-meta">${isUser ? 'You' : 'Advisor'} · ${time}${branchMark ? ' · ' + branchMark : ''}</div>`;

    let body = n.content_html || '';
    if (!/<\w+/.test(body)) body = mdToHtml(body);
    body = normalizeTrustDoc(body);

    // Branch semantics:
    //   - On a user message: "ask alternative" => new message is a sibling (same parent_id)
    //   - On an advisor message: "continue differently" => new message is a child (parent_id = advisor)
    const branchTitle  = isUser ? 'Ask an alternative (sibling branch)' : 'Continue with an alternative question';
    const branchAction = isUser ? 'alt' : 'branch';
    const actions = `
      <div class="bubble-actions">
        ${!isUser ? `
          <button class="icon-btn" data-act="copy" title="Copy response">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>` : ''}
        <button class="icon-btn" data-act="${branchAction}" title="${branchTitle}">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
        </button>
      </div>`;

    wrap.innerHTML = `
      <div class="bubble-wrap">
        ${isUser ? '' : avatar}
        <div class="bubble-content">
          ${meta}
          <div class="bubble">${body}</div>
          ${actions}
        </div>
        ${isUser ? avatar : ''}
      </div>`;

    elThread.appendChild(wrap);
  });

  // bind actions
  elThread.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const act = btn.dataset.act;
      const id  = btn.closest('.msg').dataset.id;
      if (act === 'copy')  { copyMessage(id) }
      if (act === 'branch'){ setBranchFrom(id, 'child') }       // next msg parent = this advisor
      if (act === 'alt')   { setBranchFrom(id, 'sibling') }     // next msg parent = this user's parent
    });
  });

  // scroll to bottom on fresh render
  requestAnimationFrame(() => { elThreadS.scrollTop = elThreadS.scrollHeight });
}

function welcomeHTML(){
  const suggestions = [
    { title: 'Draft a trustee resolution',
      desc: 'For a quarterly distribution under a non-grantor trust.',
      prompt: 'Draft a trustee resolution authorizing a quarterly distribution under a non-grantor irrevocable trust.' },
    { title: '§643(b) DNI computation',
      desc: 'Walk through DNI calculation for a complex trust.',
      prompt: 'Walk through §643(b) DNI computation for a complex non-grantor trust with capital gains retained in corpus.' },
    { title: '508(c)(1)(A) defense package',
      desc: 'Outline the 14-point IRS church test documentation.',
      prompt: 'Outline the 14-point IRS church test and the evidence needed for each element in a 508(c)(1)(A) defense package.' },
    { title: 'Grantor trust risk review',
      desc: 'Identify clauses that trigger §§671–679 treatment.',
      prompt: 'Review the typical clauses that risk triggering grantor trust treatment under IRC §§671–679 and how to draft around them.' },
  ];
  const cards = suggestions.map(s =>
    `<button class="suggestion" data-prompt="${escapeHtml(s.prompt)}">
       <div class="title">${s.title}</div>
       <div class="desc">${s.desc}</div>
     </button>`).join('');
  return `
    <div class="welcome">
      <div class="welcome-title">Private Trust Fiduciary Advisor</div>
      <div class="welcome-sub">Formal trust, fiduciary, and contractual analysis with strategic guidance, grounded in your private knowledge base.</div>
      <div class="suggestions">${cards}</div>
    </div>`;
}
function bindSuggestions(){
  document.querySelectorAll('.suggestion').forEach(b => {
    b.addEventListener('click', () => {
      elInput.textContent = b.dataset.prompt;
      elInput.focus();
      placeCaretAtEnd(elInput);
    });
  });
}

/* ===================== Tree rendering ===================== */
function renderTree(){
  if (!state.currentChatId){
    elTreeBody.innerHTML = '<div class="tree-empty">Select or start a conversation to see its tree.</div>';
    return;
  }
  if (!state.treeNodes.length){
    elTreeBody.innerHTML = '<div class="tree-empty">No messages yet. Send a question to begin.</div>';
    return;
  }

  // Build children map
  const byParent = {};
  state.treeNodes.forEach(n => {
    const p = n.parent_id || 'ROOT';
    (byParent[p] = byParent[p] || []).push(n);
  });
  Object.values(byParent).forEach(arr =>
    arr.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
  );

  const path = new Set(pathFromLeaf(state.activeLeafId).map(n => n.id));

  const out = [];
  function walk(parent, depth){
    (byParent[parent] || []).forEach(n => {
      const isActive = n.id === state.activeLeafId;
      const inPath = path.has(n.id);
      const label = (() => {
        const t = plainText(n.content_html);
        if (!t) return n.role === 'user' ? 'User message' : 'Advisor response';
        return t.length > 80 ? t.slice(0, 80) + '…' : t;
      })();
      const classes = ['tree-node', n.role === 'user' ? 'user' : 'assistant'];
      if (isActive) classes.push('active');
      if (inPath)   classes.push('in-path');
      const role = n.role === 'user' ? 'U' : 'A';
      out.push(`
        <div class="${classes.join(' ')}" data-id="${n.id}" data-depth="${depth}">
          <span class="dot"></span>
          <div class="tree-row">
            <span class="role-tag">${role}</span>
            <span class="tree-label" title="${escapeHtml(plainText(n.content_html))}">${escapeHtml(label)}</span>
            <div class="tree-actions">
              <button class="icon-btn" data-tact="open" title="Open this path">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/></svg>
              </button>
              ${n.role === 'user' ? `
              <button class="icon-btn" data-tact="branch" title="Branch from here">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>
              </button>` : ''}
            </div>
          </div>
        </div>`);
      walk(n.id, depth + 1);
    });
  }
  walk('ROOT', 0);

  elTreeBody.innerHTML = out.join('');

  elTreeBody.querySelectorAll('.tree-node').forEach(el => {
    el.addEventListener('click', () => {
      state.activeLeafId = findDescendantLeaf(el.dataset.id);
      renderThread();
      renderTree();
    });
  });
  elTreeBody.querySelectorAll('[data-tact]').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const act = btn.dataset.tact;
      const id  = btn.closest('.tree-node').dataset.id;
      if (act === 'open'){
        state.activeLeafId = findDescendantLeaf(id);
        renderThread();
        renderTree();
      }
      if (act === 'branch'){ setBranchFrom(id) }
    });
  });
}

function findDescendantLeaf(startId){
  // Prefer the most recent descendant of startId; fallback to startId itself.
  const children = {};
  state.treeNodes.forEach(n => {
    const p = n.parent_id;
    if (p) (children[p] = children[p] || []).push(n);
  });
  let cur = startId;
  while (children[cur] && children[cur].length){
    children[cur].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    cur = children[cur][0].id;
  }
  return cur;
}

/* ===================== Branching =====================
 * Two modes:
 *   'child'   -> next user message becomes a CHILD of the clicked node
 *                (typical: continue down an advisor's answer with a new
 *                 alternative follow-up question).
 *   'sibling' -> next user message becomes a SIBLING of the clicked node
 *                (typical: ask an alternative version of a user question
 *                 at the same conversation level — shares the same parent).
 */
function setBranchFrom(msgId, mode){
  const n = state.treeNodes.find(x => x.id === msgId);
  if (!n) return;
  mode = mode || 'child';

  let parentId = null;
  let labelPrefix = '';
  if (mode === 'sibling'){
    // Sibling: parent = clicked node's parent (may be null for root-level)
    parentId = n.parent_id || null;
    labelPrefix = 'Asking alternative to: ';
  } else {
    // Child: parent = clicked node
    parentId = n.id;
    labelPrefix = 'Branching from: ';
  }

  state.branchFromId = parentId;
  const txt = plainText(n.content_html);
  const snippet = txt.length > 80 ? txt.slice(0, 80) + '…' : (txt || '(root)');
  elBranchText.textContent = labelPrefix + snippet;
  elBranchInd.classList.add('show');
  elInput.focus();
}
function clearBranch(){
  state.branchFromId = null;
  elBranchInd.classList.remove('show');
}
document.getElementById('branch-clear').addEventListener('click', clearBranch);

/* ===================== Graph View =====================
 * An interactive 2D canvas for the conversation tree.
 *
 * Features:
 *   - Pan (drag background) / Zoom (wheel, pinch, buttons)
 *   - Drag nodes to manually position them (persisted to backend)
 *   - Double-click empty canvas to create a new 'topic' node
 *   - Zone tool: click-drag to draw a colored region around nodes
 *   - Click node/zone to open narrative side panel with editable details
 *   - Re-layout button to clear manual positions and snap back to auto-layout
 *
 * Layout rules:
 *   - Nodes with manually_positioned=1 use graph_x/graph_y
 *   - Other nodes get tidy-tree auto-layout, positioned relative to their
 *     ancestors' manual positions where those exist
 */
const graphState = {
  transform: { x: 40, y: 40, k: 1 },
  drag: null,             // { type: 'pan'|'node'|'zone-draw'|'zone-move'|'zone-resize', ... }
  svg: null,
  stage: null,
  layout: null,
  mode: 'select',         // 'select' | 'zone' | 'topic'
  selected: null,         // { kind: 'node'|'zone', id }
  zones: [],              // loaded from /tree
  _drewSinceDown: false,
  _pendingClick: null,
};

const GRAPH_NODE_W = 220;
const GRAPH_NODE_H = 66;
const GRAPH_ROW_H  = 130;
const GRAPH_COL_W  = 250;
const DRAG_THRESHOLD = 4;   // px before a click becomes a drag

const ZONE_PALETTE = [
  '#8b6f3e', '#3e6b8b', '#6b3e8b', '#8b3e52',
  '#3e8b6f', '#8b853e', '#555b6e', '#a0522d',
];

function setGraphMode(mode){
  graphState.mode = mode;
  document.querySelectorAll('[data-graph-tool]').forEach(el => {
    el.classList.toggle('active', el.dataset.graphTool === mode);
  });
  const stage = graphState.stage;
  if (stage){
    stage.classList.toggle('mode-zone', mode === 'zone');
    stage.classList.toggle('mode-topic', mode === 'topic');
    stage.classList.toggle('mode-select', mode === 'select');
  }
}

function openGraphModal(){
  const root = document.getElementById('graph-root');
  root.classList.add('show');
  requestAnimationFrame(() => renderGraph(true));
}
function closeGraphModal(){
  document.getElementById('graph-root').classList.remove('show');
  closeNarrativePanel();
}
document.getElementById('btn-open-graph').addEventListener('click', openGraphModal);
document.getElementById('graph-close').addEventListener('click', closeGraphModal);

// Tool buttons
document.querySelectorAll('[data-graph-tool]').forEach(btn => {
  btn.addEventListener('click', () => setGraphMode(btn.dataset.graphTool));
});

document.getElementById('graph-relayout').addEventListener('click', async () => {
  const ok = await modal({
    title: 'Reset to auto-layout?',
    desc: 'This clears every manual position in this chat and re-runs the tidy-tree algorithm. Zones are kept.',
    okLabel: 'Reset layout',
  });
  if (!ok) return;
  if (!state.currentChatId) return;
  try {
    await api('POST', '/chats/' + state.currentChatId + '/relayout', { form: new FormData() });
    // clear local flags
    state.treeNodes.forEach(n => {
      n.manually_positioned = 0;
      n.graph_x = null; n.graph_y = null;
    });
    renderGraph(true);
    toast('Layout reset', 'success');
  } catch(e){
    toast('Relayout failed: ' + e.message, 'error');
  }
});

/* ---------- Layout ---------- */
function computeGraphLayout(){
  const nodes = state.treeNodes;
  if (!nodes.length && !graphState.zones.length){
    return { nodes: [], edges: [], branchLabels: [], bounds: null };
  }

  const byId = {};
  nodes.forEach(n => { byId[n.id] = n });
  const childrenMap = {};
  const roots = [];
  nodes.forEach(n => {
    if (n.parent_id && byId[n.parent_id]){
      (childrenMap[n.parent_id] = childrenMap[n.parent_id] || []).push(n);
    } else {
      roots.push(n);
    }
  });
  Object.values(childrenMap).forEach(arr =>
    arr.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
  );
  roots.sort((a, b) => new Date(a.created_at) - new Date(b.created_at));

  const subW = {};
  function measure(id){
    const kids = childrenMap[id] || [];
    if (!kids.length){ subW[id] = 1; return 1 }
    let w = 0;
    kids.forEach(k => { w += measure(k.id) });
    subW[id] = w;
    return w;
  }
  roots.forEach(r => measure(r.id));

  // Build active-path set for highlighting
  const inPath = new Set();
  let cur = state.activeLeafId;
  while (cur && byId[cur]){
    inPath.add(cur);
    cur = byId[cur].parent_id;
  }

  // Auto-layout positions (fallback for nodes without manual position)
  const autoPos = {};
  let xCursor = 0;
  function place(id, left, depth){
    const w = subW[id];
    const cx = (left + w / 2) * GRAPH_COL_W;
    const cy = depth * GRAPH_ROW_H;
    autoPos[id] = { x: cx, y: cy, depth };
    const kids = childrenMap[id] || [];
    let off = left;
    kids.forEach(k => { place(k.id, off, depth + 1); off += subW[k.id] });
  }
  roots.forEach(r => { place(r.id, xCursor, 0); xCursor += subW[r.id] });

  // Final positions: manual > auto
  const pos = {};
  nodes.forEach(n => {
    if (n.manually_positioned && n.graph_x != null && n.graph_y != null){
      pos[n.id] = { x: n.graph_x, y: n.graph_y };
    } else if (autoPos[n.id]){
      pos[n.id] = { x: autoPos[n.id].x, y: autoPos[n.id].y };
    }
  });

  // Edges
  const edges = [];
  nodes.forEach(n => {
    if (n.parent_id && pos[n.parent_id] && pos[n.id]){
      const p = pos[n.parent_id], c = pos[n.id];
      edges.push({
        from: n.parent_id, to: n.id,
        x1: p.x, y1: p.y + GRAPH_NODE_H / 2,
        x2: c.x, y2: c.y - GRAPH_NODE_H / 2,
        inPath: inPath.has(n.parent_id) && inPath.has(n.id),
      });
    }
  });

  // Branch labels (first user/topic under a fork point)
  const branchLabels = [];
  nodes.forEach(n => {
    if (!n.parent_id) return;
    const sibs = childrenMap[n.parent_id] || [];
    if (sibs.length <= 1) return;
    if (n.role !== 'user' && n.role !== 'topic') return;
    const label = n.custom_title
      ? n.custom_title
      : plainText(n.content_html).trim();
    if (!label) return;
    const short = label.length > 26 ? label.slice(0, 26) + '…' : label;
    const p = pos[n.id];
    if (p) branchLabels.push({ x: p.x, y: p.y - GRAPH_NODE_H / 2 - 14, label: short });
  });

  // Bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const laidOut = nodes
    .filter(n => pos[n.id])
    .map(n => {
      const p = pos[n.id];
      minX = Math.min(minX, p.x - GRAPH_NODE_W / 2);
      maxX = Math.max(maxX, p.x + GRAPH_NODE_W / 2);
      minY = Math.min(minY, p.y - GRAPH_NODE_H / 2);
      maxY = Math.max(maxY, p.y + GRAPH_NODE_H / 2);
      return {
        id: n.id, x: p.x, y: p.y,
        msg: n,
        inPath: inPath.has(n.id),
        active: n.id === state.activeLeafId,
        selected: graphState.selected &&
                  graphState.selected.kind === 'node' &&
                  graphState.selected.id === n.id,
      };
    });

  graphState.zones.forEach(z => {
    minX = Math.min(minX, z.x);
    maxX = Math.max(maxX, z.x + z.w);
    minY = Math.min(minY, z.y);
    maxY = Math.max(maxY, z.y + z.h);
  });

  if (!isFinite(minX)){ minX = 0; maxX = 400; minY = 0; maxY = 400 }

  return {
    nodes: laidOut, edges, branchLabels,
    bounds: { minX, maxX, minY, maxY,
              width: maxX - minX, height: maxY - minY },
  };
}

/* ---------- Render ---------- */
function renderGraph(fitView){
  const svg = document.getElementById('graph-svg');
  const stage = document.getElementById('graph-stage');
  const empty = document.getElementById('graph-empty');
  graphState.svg = svg;
  graphState.stage = stage;
  setGraphMode(graphState.mode);

  const layout = computeGraphLayout();
  graphState.layout = layout;

  const nothing = !layout.nodes.length && !graphState.zones.length;
  empty.classList.toggle('hide', !nothing);

  const xmlns = 'http://www.w3.org/2000/svg';
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const rootG = document.createElementNS(xmlns, 'g');
  rootG.setAttribute('id', 'graph-root-g');
  svg.appendChild(rootG);

  // Zones (drawn first, under everything)
  graphState.zones.forEach(z => {
    const g = document.createElementNS(xmlns, 'g');
    const isSelected = graphState.selected &&
                       graphState.selected.kind === 'zone' &&
                       graphState.selected.id === z.id;
    g.setAttribute('class', 'graph-zone' + (isSelected ? ' selected' : ''));
    g.setAttribute('data-zone-id', z.id);

    const rect = document.createElementNS(xmlns, 'rect');
    rect.setAttribute('x', z.x); rect.setAttribute('y', z.y);
    rect.setAttribute('width', z.w); rect.setAttribute('height', z.h);
    rect.setAttribute('rx', 14);
    rect.setAttribute('fill', z.color || '#8b6f3e');
    rect.setAttribute('fill-opacity', '0.10');
    rect.setAttribute('stroke', z.color || '#8b6f3e');
    rect.setAttribute('stroke-width', isSelected ? 2 : 1.5);
    rect.setAttribute('stroke-dasharray', isSelected ? '' : '6 4');
    g.appendChild(rect);

    const t = document.createElementNS(xmlns, 'text');
    t.setAttribute('x', z.x + 14);
    t.setAttribute('y', z.y + 22);
    t.setAttribute('class', 'zone-title');
    t.setAttribute('fill', z.color || '#8b6f3e');
    t.textContent = (z.title || 'Untitled zone').toUpperCase();
    g.appendChild(t);

    if (isSelected){
      // Resize handle (bottom-right)
      const handle = document.createElementNS(xmlns, 'rect');
      handle.setAttribute('x', z.x + z.w - 10);
      handle.setAttribute('y', z.y + z.h - 10);
      handle.setAttribute('width', 12);
      handle.setAttribute('height', 12);
      handle.setAttribute('class', 'zone-resize');
      handle.setAttribute('fill', z.color || '#8b6f3e');
      g.appendChild(handle);
    }

    rootG.appendChild(g);
  });

  // Edges
  layout.edges.forEach(e => {
    const path = document.createElementNS(xmlns, 'path');
    const midY = (e.y1 + e.y2) / 2;
    const d = 'M ' + e.x1 + ' ' + e.y1 +
              ' C ' + e.x1 + ' ' + midY + ', ' +
                      e.x2 + ' ' + midY + ', ' +
                      e.x2 + ' ' + e.y2;
    path.setAttribute('d', d);
    path.setAttribute('class', 'graph-edge' + (e.inPath ? ' in-path' : ''));
    rootG.appendChild(path);
  });

  // Branch labels
  layout.branchLabels.forEach(bl => {
    const t = document.createElementNS(xmlns, 'text');
    t.setAttribute('x', bl.x);
    t.setAttribute('y', bl.y);
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('class', 'graph-branch-label');
    t.textContent = bl.label.toUpperCase();
    rootG.appendChild(t);
  });

  // Nodes
  layout.nodes.forEach(n => {
    const g = document.createElementNS(xmlns, 'g');
    const role = n.msg.role || 'user';
    const klass = 'graph-node ' + role
      + (n.inPath ? ' in-path' : '')
      + (n.active ? ' active' : '')
      + (n.selected ? ' selected' : '')
      + (n.msg.manually_positioned ? ' pinned' : '');
    g.setAttribute('class', klass);
    g.setAttribute('transform', 'translate(' + (n.x - GRAPH_NODE_W / 2) + ',' + (n.y - GRAPH_NODE_H / 2) + ')');
    g.setAttribute('data-id', n.id);

    const rect = document.createElementNS(xmlns, 'rect');
    rect.setAttribute('class', 'node-bg');
    rect.setAttribute('x', 0); rect.setAttribute('y', 0);
    rect.setAttribute('width', GRAPH_NODE_W);
    rect.setAttribute('height', GRAPH_NODE_H);
    rect.setAttribute('rx', 10);
    g.appendChild(rect);

    const dot = document.createElementNS(xmlns, 'circle');
    dot.setAttribute('class', 'node-dot');
    dot.setAttribute('cx', 14); dot.setAttribute('cy', 14); dot.setAttribute('r', 3.5);
    g.appendChild(dot);

    const roleLabel = document.createElementNS(xmlns, 'text');
    roleLabel.setAttribute('class', 'node-role');
    roleLabel.setAttribute('x', 26); roleLabel.setAttribute('y', 17);
    const roleText = role === 'topic' ? 'TOPIC'
                    : role === 'advisor' ? 'ADVISOR' : 'YOU';
    roleLabel.textContent = roleText;
    g.appendChild(roleLabel);

    // Pinned indicator
    if (n.msg.manually_positioned){
      const pin = document.createElementNS(xmlns, 'circle');
      pin.setAttribute('cx', GRAPH_NODE_W - 14);
      pin.setAttribute('cy', 14);
      pin.setAttribute('r', 3);
      pin.setAttribute('class', 'node-pin');
      g.appendChild(pin);
    }

    const label = document.createElementNS(xmlns, 'text');
    label.setAttribute('class', 'node-label');
    label.setAttribute('x', 14); label.setAttribute('y', 40);
    const displayText = n.msg.custom_title
      ? n.msg.custom_title
      : plainText(n.msg.content_html).replace(/\s+/g, ' ').trim();
    label.textContent = displayText.length > 32
      ? displayText.slice(0, 32) + '…'
      : (displayText || '(empty)');
    g.appendChild(label);

    // Narrative badge
    if (n.msg.narrative && n.msg.narrative.trim()){
      const bg = document.createElementNS(xmlns, 'circle');
      bg.setAttribute('cx', GRAPH_NODE_W - 14);
      bg.setAttribute('cy', GRAPH_NODE_H - 14);
      bg.setAttribute('r', 6);
      bg.setAttribute('class', 'node-narrative-badge');
      g.appendChild(bg);
    }

    rootG.appendChild(g);
  });

  applyGraphTransform();
  if (fitView) fitGraphToView();
}

function applyGraphTransform(){
  const g = document.getElementById('graph-root-g');
  if (!g) return;
  const t = graphState.transform;
  g.setAttribute('transform', 'translate(' + t.x + ',' + t.y + ') scale(' + t.k + ')');
}

function fitGraphToView(){
  const L = graphState.layout;
  if (!L || !L.bounds) return;
  const stage = graphState.stage;
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const pad = 60;
  const bw = L.bounds.width + pad * 2;
  const bh = L.bounds.height + pad * 2;
  if (bw <= 0 || bh <= 0){ graphState.transform = { x: 40, y: 40, k: 1 }; applyGraphTransform(); return }
  const k = Math.min(sw / bw, sh / bh, 1.2);
  graphState.transform.k = k;
  graphState.transform.x = -L.bounds.minX * k + (sw - L.bounds.width * k) / 2;
  graphState.transform.y = -L.bounds.minY * k + (sh - L.bounds.height * k) / 2;
  applyGraphTransform();
}

/* ---------- Coordinate helpers ---------- */
function screenToWorld(clientX, clientY){
  const stage = graphState.stage;
  const r = stage.getBoundingClientRect();
  const t = graphState.transform;
  return {
    x: (clientX - r.left - t.x) / t.k,
    y: (clientY - r.top  - t.y) / t.k,
  };
}

/* ---------- Interaction ---------- */
(function bindGraphInteraction(){
  const stage = document.getElementById('graph-stage');
  const svg   = document.getElementById('graph-svg');

  // Mouse down
  stage.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return;
    const nodeEl = e.target.closest('.graph-node');
    const zoneEl = e.target.closest('.graph-zone');
    const resizeEl = e.target.closest('.zone-resize');
    const start = { cx: e.clientX, cy: e.clientY };
    graphState._drewSinceDown = false;

    if (resizeEl && graphState.selected && graphState.selected.kind === 'zone'){
      const z = graphState.zones.find(x => x.id === graphState.selected.id);
      if (z){
        graphState.drag = {
          type: 'zone-resize', id: z.id, start,
          orig: { x: z.x, y: z.y, w: z.w, h: z.h },
        };
        e.preventDefault(); return;
      }
    }

    if (nodeEl && graphState.mode === 'select'){
      const id = nodeEl.dataset.id;
      const node = state.treeNodes.find(x => x.id === id);
      if (node){
        const wp = screenToWorld(e.clientX, e.clientY);
        const cur = graphState.layout.nodes.find(p => p.id === id);
        graphState.drag = {
          type: 'node', id, start,
          orig: cur ? { x: cur.x, y: cur.y } : { x: 0, y: 0 },
          offset: cur ? { x: wp.x - cur.x, y: wp.y - cur.y } : { x: 0, y: 0 },
        };
        e.preventDefault(); return;
      }
    }

    if (zoneEl && graphState.mode === 'select'){
      const id = zoneEl.dataset.zoneId;
      const z = graphState.zones.find(x => x.id === id);
      if (z){
        graphState.drag = {
          type: 'zone-move', id, start,
          orig: { x: z.x, y: z.y },
        };
        e.preventDefault(); return;
      }
    }

    if (graphState.mode === 'zone'){
      const wp = screenToWorld(e.clientX, e.clientY);
      graphState.drag = {
        type: 'zone-draw', start, origin: wp,
        current: { x: wp.x, y: wp.y, w: 0, h: 0 },
      };
      e.preventDefault(); return;
    }

    // Default: pan
    graphState.drag = {
      type: 'pan', start,
      tx: graphState.transform.x,
      ty: graphState.transform.y,
    };
    stage.classList.add('dragging');
  });

  window.addEventListener('mousemove', (e) => {
    const d = graphState.drag;
    if (!d) return;
    const dx = e.clientX - d.start.cx;
    const dy = e.clientY - d.start.cy;
    if (Math.abs(dx) > DRAG_THRESHOLD || Math.abs(dy) > DRAG_THRESHOLD){
      graphState._drewSinceDown = true;
    }

    if (d.type === 'pan'){
      graphState.transform.x = d.tx + dx;
      graphState.transform.y = d.ty + dy;
      applyGraphTransform();

    } else if (d.type === 'node'){
      const wp = screenToWorld(e.clientX, e.clientY);
      const nx = wp.x - d.offset.x;
      const ny = wp.y - d.offset.y;
      // Update the SVG group live
      const g = svg.querySelector('.graph-node[data-id="' + d.id + '"]');
      if (g){
        g.setAttribute('transform',
          'translate(' + (nx - GRAPH_NODE_W / 2) + ',' + (ny - GRAPH_NODE_H / 2) + ')');
        g.classList.add('dragging');
      }
      // Update edges live
      const layoutNode = graphState.layout.nodes.find(p => p.id === d.id);
      if (layoutNode){ layoutNode.x = nx; layoutNode.y = ny }
      updateLiveEdges(d.id);

    } else if (d.type === 'zone-move'){
      const z = graphState.zones.find(x => x.id === d.id);
      if (!z) return;
      const scale = graphState.transform.k;
      z.x = d.orig.x + dx / scale;
      z.y = d.orig.y + dy / scale;
      const zEl = svg.querySelector('.graph-zone[data-zone-id="' + d.id + '"] rect');
      const zT  = svg.querySelector('.graph-zone[data-zone-id="' + d.id + '"] text');
      const zH  = svg.querySelector('.graph-zone[data-zone-id="' + d.id + '"] .zone-resize');
      if (zEl){ zEl.setAttribute('x', z.x); zEl.setAttribute('y', z.y) }
      if (zT){ zT.setAttribute('x', z.x + 14); zT.setAttribute('y', z.y + 22) }
      if (zH){ zH.setAttribute('x', z.x + z.w - 10); zH.setAttribute('y', z.y + z.h - 10) }

    } else if (d.type === 'zone-resize'){
      const z = graphState.zones.find(x => x.id === d.id);
      if (!z) return;
      const scale = graphState.transform.k;
      z.w = Math.max(60, d.orig.w + dx / scale);
      z.h = Math.max(40, d.orig.h + dy / scale);
      const zEl = svg.querySelector('.graph-zone[data-zone-id="' + d.id + '"] rect');
      const zH  = svg.querySelector('.graph-zone[data-zone-id="' + d.id + '"] .zone-resize');
      if (zEl){ zEl.setAttribute('width', z.w); zEl.setAttribute('height', z.h) }
      if (zH){ zH.setAttribute('x', z.x + z.w - 10); zH.setAttribute('y', z.y + z.h - 10) }

    } else if (d.type === 'zone-draw'){
      const wp = screenToWorld(e.clientX, e.clientY);
      d.current = {
        x: Math.min(d.origin.x, wp.x),
        y: Math.min(d.origin.y, wp.y),
        w: Math.abs(wp.x - d.origin.x),
        h: Math.abs(wp.y - d.origin.y),
      };
      drawGhostZone(d.current);
    }
  });

  window.addEventListener('mouseup', async (e) => {
    const d = graphState.drag;
    graphState.drag = null;
    stage.classList.remove('dragging');
    if (!d) return;

    if (d.type === 'node'){
      const g = svg.querySelector('.graph-node[data-id="' + d.id + '"]');
      if (g) g.classList.remove('dragging');
      if (!graphState._drewSinceDown){
        // treated as click
        handleNodeClick(d.id, e);
        return;
      }
      // Persist new position
      const layoutNode = graphState.layout.nodes.find(p => p.id === d.id);
      const node = state.treeNodes.find(x => x.id === d.id);
      if (!layoutNode || !node) return;
      node.graph_x = layoutNode.x;
      node.graph_y = layoutNode.y;
      node.manually_positioned = 1;
      try {
        const fd = new FormData();
        fd.append('x', layoutNode.x);
        fd.append('y', layoutNode.y);
        fd.append('manual', '1');
        await api('POST', '/messages/' + d.id + '/position', { form: fd });
      } catch(err){
        toast('Save position failed: ' + err.message, 'error');
      }
      renderGraph(false);

    } else if (d.type === 'zone-move'){
      if (!graphState._drewSinceDown){
        handleZoneClick(d.id, e); return;
      }
      const z = graphState.zones.find(x => x.id === d.id);
      if (!z) return;
      try {
        const fd = new FormData();
        fd.append('x', z.x); fd.append('y', z.y);
        await api('POST', '/zones/' + d.id, { form: fd });
      } catch(err){
        toast('Move zone failed: ' + err.message, 'error');
      }

    } else if (d.type === 'zone-resize'){
      const z = graphState.zones.find(x => x.id === d.id);
      if (!z) return;
      try {
        const fd = new FormData();
        fd.append('w', z.w); fd.append('h', z.h);
        await api('POST', '/zones/' + d.id, { form: fd });
      } catch(err){
        toast('Resize zone failed: ' + err.message, 'error');
      }
      renderGraph(false);

    } else if (d.type === 'zone-draw'){
      removeGhostZone();
      const c = d.current;
      if (c.w < 40 || c.h < 30){ return }  // too small, discard
      const title = await modal({
        title: 'Name this zone',
        desc: 'E.g., Real Estate, Trust Governance, Tax Strategy.',
        placeholder: 'Zone name',
      });
      if (title === null){ return }  // cancelled
      const color = ZONE_PALETTE[graphState.zones.length % ZONE_PALETTE.length];
      try {
        const fd = new FormData();
        fd.append('title', title || 'Untitled zone');
        fd.append('color', color);
        fd.append('x', c.x); fd.append('y', c.y);
        fd.append('w', c.w); fd.append('h', c.h);
        const res = await api('POST', '/chats/' + state.currentChatId + '/zones', { form: fd });
        graphState.zones.push({
          id: res.id, title: title || 'Untitled zone', narrative: null,
          color, x: c.x, y: c.y, w: c.w, h: c.h,
        });
        setGraphMode('select');
        renderGraph(false);
        toast('Zone created', 'success');
      } catch(err){
        toast('Create zone failed: ' + err.message, 'error');
      }
    }
  });

  // Zoom
  stage.addEventListener('wheel', (e) => {
    e.preventDefault();
    const t = graphState.transform;
    const delta = -e.deltaY * 0.0015;
    const newK = Math.max(0.2, Math.min(3, t.k * (1 + delta)));
    const rect = stage.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    t.x = mx - (mx - t.x) * (newK / t.k);
    t.y = my - (my - t.y) * (newK / t.k);
    t.k = newK;
    applyGraphTransform();
  }, { passive: false });

  // Double-click empty space -> create topic node
  stage.addEventListener('dblclick', async (e) => {
    if (e.target.closest('.graph-node') || e.target.closest('.graph-zone')) return;
    if (!state.currentChatId){
      toast('Start a chat first', 'info'); return;
    }
    const wp = screenToWorld(e.clientX, e.clientY);
    const title = await modal({
      title: 'New topic',
      desc: 'Creates a container node on the canvas. Send a message later to attach real content under it.',
      placeholder: 'e.g., Real Estate Holdings',
    });
    if (title === null) return;
    try {
      const fd = new FormData();
      fd.append('title', title || 'Untitled topic');
      fd.append('x', wp.x); fd.append('y', wp.y);
      const res = await api('POST', '/chats/' + state.currentChatId + '/topic', { form: fd });
      // Add to local tree
      state.treeNodes.push({
        id: res.id, role: 'topic',
        content_html: '<p><strong>' + escapeHtml(title || 'Untitled topic') + '</strong></p>',
        content_raw: title || 'Untitled topic',
        created_at: res.created_at, parent_id: null,
        graph_x: wp.x, graph_y: wp.y, manually_positioned: 1,
        custom_title: title || 'Untitled topic', narrative: null,
      });
      renderGraph(false);
      toast('Topic created', 'success');
    } catch(err){
      toast('Create topic failed: ' + err.message, 'error');
    }
  });

  // Touch
  let touchStart = null;
  stage.addEventListener('touchstart', (e) => {
    if (e.touches.length !== 1) return;
    if (e.target.closest('.graph-node') || e.target.closest('.graph-zone')) return;
    touchStart = {
      x: e.touches[0].clientX, y: e.touches[0].clientY,
      tx: graphState.transform.x, ty: graphState.transform.y,
    };
  }, { passive: true });
  stage.addEventListener('touchmove', (e) => {
    if (!touchStart || e.touches.length !== 1) return;
    graphState.transform.x = touchStart.tx + (e.touches[0].clientX - touchStart.x);
    graphState.transform.y = touchStart.ty + (e.touches[0].clientY - touchStart.y);
    applyGraphTransform();
  }, { passive: true });
  stage.addEventListener('touchend', () => { touchStart = null });
})();

function updateLiveEdges(nodeId){
  const svg = graphState.svg;
  const ns = 'http://www.w3.org/2000/svg';
  const L = graphState.layout;
  // Recompute paths touching nodeId
  L.edges.forEach((e, i) => {
    if (e.from !== nodeId && e.to !== nodeId) return;
    const p = L.nodes.find(x => x.id === e.from);
    const c = L.nodes.find(x => x.id === e.to);
    if (!p || !c) return;
    e.x1 = p.x; e.y1 = p.y + GRAPH_NODE_H / 2;
    e.x2 = c.x; e.y2 = c.y - GRAPH_NODE_H / 2;
    // Find the path SVG node — they're children of rootG in edge order.
    // Safer: re-serialize all edges by index.
    const edges = svg.querySelectorAll('path.graph-edge');
    const path = edges[i];
    if (path){
      const midY = (e.y1 + e.y2) / 2;
      const d = 'M ' + e.x1 + ' ' + e.y1 +
                ' C ' + e.x1 + ' ' + midY + ', ' +
                        e.x2 + ' ' + midY + ', ' +
                        e.x2 + ' ' + e.y2;
      path.setAttribute('d', d);
    }
  });
}

function drawGhostZone(c){
  const svg = graphState.svg;
  const rootG = document.getElementById('graph-root-g');
  let g = svg.querySelector('#ghost-zone');
  const ns = 'http://www.w3.org/2000/svg';
  if (!g){
    g = document.createElementNS(ns, 'rect');
    g.setAttribute('id', 'ghost-zone');
    g.setAttribute('class', 'ghost-zone');
    g.setAttribute('rx', 14);
    rootG.appendChild(g);
  }
  g.setAttribute('x', c.x); g.setAttribute('y', c.y);
  g.setAttribute('width', c.w); g.setAttribute('height', c.h);
}
function removeGhostZone(){
  const g = document.querySelector('#ghost-zone');
  if (g) g.remove();
}

/* ---------- Click handlers ---------- */
function handleNodeClick(id, e){
  if (e && e.shiftKey){
    setBranchFrom(id, 'child');
    closeGraphModal();
    toast('Branch armed. Type your follow-up.', 'info');
    return;
  }
  graphState.selected = { kind: 'node', id };
  const node = state.treeNodes.find(x => x.id === id);
  renderGraph(false);
  openNarrativePanel({ kind: 'node', node });
}

function handleZoneClick(id, e){
  graphState.selected = { kind: 'zone', id };
  const z = graphState.zones.find(x => x.id === id);
  renderGraph(false);
  openNarrativePanel({ kind: 'zone', zone: z });
}

/* ---------- Zoom buttons ---------- */
document.getElementById('graph-zoom-in').addEventListener('click', () => {
  const t = graphState.transform;
  const stage = graphState.stage;
  const cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
  const newK = Math.min(3, t.k * 1.25);
  t.x = cx - (cx - t.x) * (newK / t.k);
  t.y = cy - (cy - t.y) * (newK / t.k);
  t.k = newK;
  applyGraphTransform();
});
document.getElementById('graph-zoom-out').addEventListener('click', () => {
  const t = graphState.transform;
  const stage = graphState.stage;
  const cx = stage.clientWidth / 2, cy = stage.clientHeight / 2;
  const newK = Math.max(0.2, t.k / 1.25);
  t.x = cx - (cx - t.x) * (newK / t.k);
  t.y = cy - (cy - t.y) * (newK / t.k);
  t.k = newK;
  applyGraphTransform();
});
document.getElementById('graph-zoom-fit').addEventListener('click', fitGraphToView);

// Escape
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && document.getElementById('graph-root').classList.contains('show')){
    if (document.getElementById('narrative-panel').classList.contains('show')){
      closeNarrativePanel();
    } else {
      closeGraphModal();
    }
  }
});

/* ===================== Narrative Panel ===================== */
const narrState = { kind: null, id: null };

function openNarrativePanel({ kind, node, zone }){
  const panel = document.getElementById('narrative-panel');
  const titleInput = document.getElementById('narr-title');
  const narrInput = document.getElementById('narr-text');
  const kindLabel = document.getElementById('narr-kind');
  const deleteBtn = document.getElementById('narr-delete');
  const colorRow  = document.getElementById('narr-color-row');
  const chatBtn   = document.getElementById('narr-open-chat');

  if (kind === 'node'){
    narrState.kind = 'node'; narrState.id = node.id;
    const raw = plainText(node.content_html || '').trim();
    titleInput.value = node.custom_title || raw.slice(0, 120);
    narrInput.value = node.narrative || '';
    kindLabel.textContent = (node.role || 'NODE').toUpperCase();
    document.getElementById('narr-color-row-wrap').style.display = 'none';
    deleteBtn.style.display = node.role === 'topic' ? '' : 'none';
    chatBtn.style.display = '';
  } else {
    narrState.kind = 'zone'; narrState.id = zone.id;
    titleInput.value = zone.title || '';
    narrInput.value = zone.narrative || '';
    kindLabel.textContent = 'ZONE';
    document.getElementById('narr-color-row-wrap').style.display = '';
    deleteBtn.style.display = '';
    chatBtn.style.display = 'none';
    // Build color swatches
    colorRow.innerHTML = '';
    ZONE_PALETTE.forEach(c => {
      const s = document.createElement('button');
      s.className = 'color-sw';
      if (c === zone.color) s.classList.add('active');
      s.style.background = c;
      s.dataset.color = c;
      s.addEventListener('click', () => changeZoneColor(zone.id, c));
      colorRow.appendChild(s);
    });
  }

  panel.classList.add('show');
  setTimeout(() => titleInput.focus(), 50);
}
function closeNarrativePanel(){
  document.getElementById('narrative-panel').classList.remove('show');
  graphState.selected = null;
  narrState.kind = null; narrState.id = null;
  renderGraph(false);
}
document.getElementById('narr-close').addEventListener('click', closeNarrativePanel);

async function saveNarrative(){
  const title = document.getElementById('narr-title').value.trim();
  const narrative = document.getElementById('narr-text').value;
  if (narrState.kind === 'node'){
    const n = state.treeNodes.find(x => x.id === narrState.id);
    if (!n) return;
    n.custom_title = title || null;
    n.narrative = narrative || null;
    try {
      const fd = new FormData();
      if (title !== null) fd.append('custom_title', title);
      fd.append('narrative', narrative);
      await api('POST', '/messages/' + narrState.id + '/meta', { form: fd });
      toast('Saved', 'success');
      renderGraph(false);
    } catch(e){ toast('Save failed: ' + e.message, 'error') }
  } else if (narrState.kind === 'zone'){
    const z = graphState.zones.find(x => x.id === narrState.id);
    if (!z) return;
    z.title = title; z.narrative = narrative || null;
    try {
      const fd = new FormData();
      fd.append('title', title);
      fd.append('narrative', narrative);
      await api('POST', '/zones/' + narrState.id, { form: fd });
      toast('Saved', 'success');
      renderGraph(false);
    } catch(e){ toast('Save failed: ' + e.message, 'error') }
  }
}
document.getElementById('narr-save').addEventListener('click', saveNarrative);

async function changeZoneColor(id, color){
  const z = graphState.zones.find(x => x.id === id);
  if (!z) return;
  z.color = color;
  try {
    const fd = new FormData();
    fd.append('color', color);
    await api('POST', '/zones/' + id, { form: fd });
    document.querySelectorAll('#narr-color-row .color-sw').forEach(s => {
      s.classList.toggle('active', s.dataset.color === color);
    });
    renderGraph(false);
  } catch(e){ toast('Color update failed: ' + e.message, 'error') }
}

document.getElementById('narr-delete').addEventListener('click', async () => {
  const kind = narrState.kind, id = narrState.id;
  if (!kind || !id) return;
  const label = kind === 'zone' ? 'zone' : 'topic';
  const ok = await modal({
    title: 'Delete this ' + label + '?',
    desc: kind === 'zone'
      ? 'The zone will be removed. Nodes inside are not affected.'
      : 'The topic node will be removed. Any children are re-parented to its parent.',
    okLabel: 'Delete',
  });
  if (!ok) return;
  try {
    if (kind === 'zone'){
      await api('DELETE', '/zones/' + id);
      graphState.zones = graphState.zones.filter(z => z.id !== id);
    } else {
      await api('DELETE', '/messages/' + id);
      state.treeNodes = state.treeNodes.filter(n => n.id !== id);
      // Re-parent children locally
      state.treeNodes.forEach(n => {
        if (n.parent_id === id){
          n.parent_id = null;
        }
      });
    }
    closeNarrativePanel();
    renderGraph(false);
    toast('Deleted', 'success');
  } catch(e){ toast('Delete failed: ' + e.message, 'error') }
});

document.getElementById('narr-open-chat').addEventListener('click', () => {
  if (narrState.kind !== 'node' || !narrState.id) return;
  const n = state.treeNodes.find(x => x.id === narrState.id);
  if (!n) return;
  state.activeLeafId = findDescendantLeaf(n.id);
  setBranchFrom(n.id, 'child');
  renderThread();
  renderTree();
  closeGraphModal();
  elInput.focus();
});

/* ===================== Copy ===================== */
function copyMessage(id){
  const n = state.treeNodes.find(x => x.id === id);
  if (!n) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = n.content_html || '';
  const text = tmp.textContent || '';
  navigator.clipboard.writeText(text).then(
    () => toast('Copied to clipboard', 'success'),
    () => toast('Copy failed', 'error')
  );
}

/* ===================== Profile ===================== */
document.getElementById('pf-save').addEventListener('click', () => {
  const uid = elPfId.value.trim();
  const em  = elPfEmail.value.trim();
  if (!uid){ elPfMsg.textContent = 'Client ID is required.'; elPfMsg.className = 'msg-line error'; return }
  state.userId = uid; state.email = em;
  localStorage.setItem(LS.USER, uid);
  localStorage.setItem(LS.EMAIL, em);
  elPfMsg.textContent = 'Profile saved.';
  elPfMsg.className = 'msg-line success';
  state.currentChatId = null;
  state.treeNodes = []; graphState.zones = [];
  state.activeLeafId = null;
  renderThread();
  renderTree();
  loadChats();
});
document.getElementById('pf-logout').addEventListener('click', () => {
  localStorage.removeItem(LS.USER);
  localStorage.removeItem(LS.EMAIL);
  state.userId = ''; state.email = '';
  elPfId.value = ''; elPfEmail.value = '';
  elPfMsg.textContent = 'Logged out.';
  elPfMsg.className = 'msg-line';
  state.currentChatId = null;
  state.treeNodes = []; graphState.zones = [];
  state.activeLeafId = null;
  elChatList.innerHTML = '<div class="empty-state">Sign in with your Client ID to begin.</div>';
  renderThread();
  renderTree();
});

/* ===================== Chats ===================== */
async function loadChats(){
  elChatList.innerHTML = '';
  if (!state.userId){
    elChatList.innerHTML = '<div class="empty-state">Sign in to view conversations.</div>';
    return;
  }
  try{
    const data = await api('GET', '/chats');
    if (!data.items || !data.items.length){
      elChatList.innerHTML = '<div class="empty-state">No conversations yet. Start a new one.</div>';
      return;
    }
    data.items.forEach(ch => {
      const row = document.createElement('div');
      row.className = 'chat-item' + (state.currentChatId === ch.id ? ' active' : '');
      row.dataset.id = ch.id;
      const when = new Date(ch.updated_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      row.innerHTML = `
        <div class="title">${escapeHtml(ch.title || 'Untitled')}</div>
        <div class="meta">${when}</div>
        <div class="actions">
          <button class="icon-btn" data-cact="rename" title="Rename">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
          </button>
          <button class="icon-btn" data-cact="delete" title="Delete">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"/></svg>
          </button>
        </div>`;
      row.addEventListener('click', () => openChat(ch.id));
      row.querySelectorAll('[data-cact]').forEach(btn => {
        btn.addEventListener('click', async e => {
          e.stopPropagation();
          const act = btn.dataset.cact;
          if (act === 'rename'){
            const title = await modal({ title: 'Rename conversation', desc: 'Enter a new title.', placeholder: 'Title', value: ch.title || '' });
            if (title){
              const fd = new FormData(); fd.append('title', title);
              try { await api('POST', `/chats/${ch.id}/title`, { form: fd }); loadChats() }
              catch(err){ toast('Rename failed: ' + err.message, 'error') }
            }
          }
          if (act === 'delete'){
            const ok = await confirmDialog({ title: 'Delete conversation?', desc: 'This cannot be undone.', okLabel: 'Delete' });
            if (ok){
              try {
                await api('DELETE', `/chats/${ch.id}`);
                if (state.currentChatId === ch.id){
                  state.currentChatId = null; state.treeNodes = []; graphState.zones = []; state.activeLeafId = null;
                  elChatTitle.innerHTML = '<span class="muted">Advisor</span>';
                  renderThread(); renderTree();
                }
                loadChats();
              } catch(err){ toast('Delete failed: ' + err.message, 'error') }
            }
          }
        });
      });
      elChatList.appendChild(row);
    });
    if (!state.currentChatId && data.items.length){
      openChat(data.items[0].id);
    }
  } catch(e){
    elChatList.innerHTML = '<div class="empty-state" style="color:var(--error)">Failed to load: ' + escapeHtml(e.message) + '</div>';
  }
}

async function openChat(chatId){
  state.currentChatId = chatId;
  state.branchFromId = null;
  clearBranch();
  document.querySelectorAll('.chat-item').forEach(x => x.classList.toggle('active', x.dataset.id === chatId));
  try{
    const [chatData, treeData] = await Promise.all([
      api('GET', `/chats/${chatId}`),
      api('GET', `/chats/${chatId}/tree`),
    ]);
    state.currentChatTitle = chatData.chat.title || 'Untitled';
    elChatTitle.innerHTML = escapeHtml(state.currentChatTitle);
    state.treeNodes = treeData.nodes || [];
    graphState.zones = treeData.zones || [];
    // Default active leaf = most recent descendant path
    const lvs = leaves();
    lvs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    state.activeLeafId = lvs.length ? lvs[0].id : null;
    renderThread();
    renderTree();
  } catch(e){
    toast('Failed to open chat: ' + e.message, 'error');
  }
}

document.getElementById('btn-newchat').addEventListener('click', async () => {
  if (!state.userId){ elPfMsg.textContent = 'Sign in first.'; elPfMsg.className = 'msg-line error'; return }
  try{
    const res = await api('POST', '/chats', { form: new FormData() });
    state.currentChatId = res.chat_id;
    state.treeNodes = []; graphState.zones = [];
    state.activeLeafId = null;
    elChatTitle.innerHTML = '<span class="muted">New chat</span>';
    renderThread(); renderTree();
    loadChats();
  } catch(e){
    toast('Failed to create chat: ' + e.message, 'error');
  }
});

/* ===================== Composer ===================== */
function placeCaretAtEnd(el){
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(false);
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
}

function readInput(){
  const tmp = elInput.cloneNode(true);
  tmp.querySelectorAll('div, br').forEach(d => {
    if (d.tagName === 'BR' || d.innerHTML === '<br>') d.outerHTML = '\n';
  });
  return tmp.innerText.replace(/\u00A0/g, ' ').trim();
}

function renderFileChips(){
  elChips.innerHTML = '';
  const files = Array.from(elFile.files || []);
  files.forEach((f, i) => {
    const c = document.createElement('span');
    c.className = 'chip';
    c.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      ${escapeHtml(f.name)}
      <button data-rm="${i}" title="Remove">&times;</button>`;
    elChips.appendChild(c);
  });
  elChips.querySelectorAll('[data-rm]').forEach(b =>
    b.addEventListener('click', () => {
      const dt = new DataTransfer();
      Array.from(elFile.files).forEach((f, i) => { if (String(i) !== b.dataset.rm) dt.items.add(f) });
      elFile.files = dt.files;
      renderFileChips();
    })
  );
}

elAttach.addEventListener('click', () => elFile.click());
elFile.addEventListener('change', renderFileChips);

async function handleSend(q){
  if (!state.userId){ toast('Sign in with your Client ID first.', 'error'); return }
  if (!q) return;

  // Ensure we have a chat
  if (!state.currentChatId){
    try{
      const res = await api('POST', '/chats', { form: new FormData() });
      state.currentChatId = res.chat_id;
      state.treeNodes = []; graphState.zones = [];
      state.activeLeafId = null;
    } catch(e){
      toast('Failed to create chat: ' + e.message, 'error'); return;
    }
  }

  // Optimistic user message
  const tempUserId = 'tmp-u-' + Math.random().toString(36).slice(2);
  const tempAdvisorId = 'tmp-a-' + Math.random().toString(36).slice(2);
  const parent = state.branchFromId || (state.activeLeafId || null);
  state.treeNodes.push({
    id: tempUserId, role: 'user',
    content_html: '<p>' + escapeHtml(q).replace(/\n/g, '<br>') + '</p>',
    created_at: new Date().toISOString(), parent_id: parent,
  });
  state.treeNodes.push({
    id: tempAdvisorId, role: 'advisor',
    content_html: '<div class="bubble thinking">Consulting the knowledge base<span class="dots"><span></span><span></span><span></span></span></div>',
    created_at: new Date().toISOString(), parent_id: tempUserId, _thinking: true,
  });
  state.activeLeafId = tempAdvisorId;
  renderThread(); renderTree();

  // auto-title first prompt
  if (!state.treeNodes.filter(n => !n._thinking && !n.id.startsWith('tmp-')).length){
    const fd = new FormData(); fd.append('title', q.slice(0, 60));
    api('POST', `/chats/${state.currentChatId}/title`, { form: fd }).catch(() => {});
  }

  try{
    const files = Array.from(elFile.files || []);
    let data;
    if (files.length){
      const fd = new FormData();
      fd.append('question', q);
      fd.append('chat_id', state.currentChatId);
      if (parent) fd.append('parent_id', parent);
      files.forEach(f => fd.append('files', f));
      data = await api('POST', '/review', { form: fd });
      elFile.value = ''; renderFileChips();
    } else {
      const url = new URL('/rag', location.origin);
      url.searchParams.set('question', q);
      url.searchParams.set('chat_id', state.currentChatId);
      if (parent) url.searchParams.set('parent_id', parent);
      url.searchParams.set('top_k', '12');
      data = await api('GET', url.toString());
    }

    // Refresh tree authoritatively
    const treeData = await api('GET', `/chats/${state.currentChatId}/tree`);
    state.treeNodes = treeData.nodes || [];
    graphState.zones = treeData.zones || [];

    // New leaf = latest message descending from our branch parent
    const newestUser = [...state.treeNodes].reverse()
      .find(n => n.role === 'user' && (n.parent_id || null) === parent);
    if (newestUser){
      const newestReply = [...state.treeNodes].reverse()
        .find(n => n.role === 'advisor' && n.parent_id === newestUser.id);
      state.activeLeafId = (newestReply || newestUser).id;
    } else {
      const lvs = leaves();
      lvs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      state.activeLeafId = lvs.length ? lvs[0].id : null;
    }

    clearBranch();
    renderThread(); renderTree();
    loadChats();
  } catch(e){
    // Remove optimistic entries
    state.treeNodes = state.treeNodes.filter(n => n.id !== tempUserId && n.id !== tempAdvisorId);
    state.activeLeafId = pathFromLeaf(state.activeLeafId).length ? state.activeLeafId : null;
    renderThread(); renderTree();
    toast('Request failed: ' + e.message, 'error');
  }
}

elInput.addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && !ev.shiftKey){
    ev.preventDefault();
    const q = readInput();
    if (!q) return;
    elInput.innerHTML = '';
    handleSend(q);
  }
});
elSend.addEventListener('click', () => {
  const q = readInput();
  if (!q) return;
  elInput.innerHTML = '';
  handleSend(q);
});

/* ===================== Layout / theme toggles ===================== */
document.getElementById('btn-rail-collapse').addEventListener('click', () => {
  ui.sidebarOpen = false; saveUI(); applyLayout();
});
document.getElementById('left-tab').addEventListener('click', () => {
  ui.sidebarOpen = true; saveUI(); applyLayout();
});
document.getElementById('btn-tree-collapse').addEventListener('click', () => {
  ui.treeOpen = false; saveUI(); applyLayout();
});
document.getElementById('right-tab').addEventListener('click', () => {
  ui.treeOpen = true; saveUI(); applyLayout();
});
document.getElementById('btn-toggle-rail-m').addEventListener('click', () => {
  ui.sidebarOpen = !ui.sidebarOpen; saveUI(); applyLayout();
});
document.getElementById('btn-toggle-tree-m').addEventListener('click', () => {
  ui.treeOpen = !ui.treeOpen; saveUI(); applyLayout();
});
document.getElementById('btn-theme').addEventListener('click', () => {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem(LS.THEME, state.theme);
  applyTheme();
});

/* Keyboard shortcuts */
document.addEventListener('keydown', e => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k'){ e.preventDefault(); elInput.focus() }
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'O'){ e.preventDefault(); document.getElementById('btn-newchat').click() }
  if ((e.metaKey || e.ctrlKey) && e.key === '\\'){ e.preventDefault(); ui.sidebarOpen = !ui.sidebarOpen; saveUI(); applyLayout() }
});

/* ===================== Init ===================== */
applyTheme();
applyLayout();

if (state.userId){ loadChats() }
else { elChatList.innerHTML = '<div class="empty-state">Sign in with your Client ID to begin.</div>' }

renderThread();
renderTree();
</script>
</body>
</html>
"""

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return HTMLResponse(WIDGET_HTML)

# ========== Health / Diag ==========
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diag")
def diag():
    info = {
        "has_PINECONE_API_KEY": bool(os.getenv("PINECONE_API_KEY")),
        "has_OPENAI_API_KEY":   bool(os.getenv("OPENAI_API_KEY")),
        "PINECONE_INDEX": index_name or None,
        "PINECONE_HOST":  host or None,
        "NO_PROXY": os.getenv("NO_PROXY"),
        "db_path": DB_PATH,
    }
    try:
        lst = pc.list_indexes()
        info["pinecone_ok"]  = True
        info["index_count"]  = len(lst or [])
    except Exception as e:
        info["pinecone_ok"] = False
        info["error"] = str(e)
    return info

# ========== /search (raw context) ==========
@app.get("/search")
def search_endpoint(
    question: str = Query(..., min_length=3),
    top_k: int = Query(12, ge=1, le=30),
    level: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    user_id: str = Depends(get_current_user),
):
    require_auth(authorization)
    check_rate_limit()
    t0 = time.time()
    try:
        emb = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
        flt = {"doc_level": {"$eq": level}} if level else None
        res = idx.query(vector=emb, top_k=max(top_k, 12), include_metadata=True, filter=flt)
        matches = res["matches"] if isinstance(res, dict) else getattr(res, "matches", [])
        uniq = _dedup_and_rank_sources(matches, top_k=top_k)
        titles = _titles_only(uniq)
        rows = []
        for s in uniq:
            meta = s.get("meta", {})
            rows.append({
                "title":   s["title"],
                "level":   s["level"],
                "page":    s["page"],
                "version": s.get("version", ""),
                "score":   s["score"],
                "snippet": _extract_snippet(meta) or "",
            })
        return {"question": question, "titles": titles, "matches": rows, "t_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ========== /rag (synthesis + persistence + branching) ==========
@app.get("/rag")
def rag_endpoint(
    question: str = Query(..., min_length=3),
    chat_id: Optional[str] = Query(None),
    parent_id: Optional[str] = Query(None),
    top_k: int = Query(12, ge=1, le=30),
    level: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
    user_id: str = Depends(get_current_user),
):
    require_auth(authorization)
    check_rate_limit()
    conn = db()
    try:
        chat_id = ensure_chat(conn, user_id, chat_id)
        # If caller didn't specify a branch point, fall back to the latest message in the chat
        effective_parent = parent_id or _latest_leaf(conn, chat_id)
        user_msg_id = insert_message(
            conn, chat_id, user_id, "user",
            content_html=f"<p>{question}</p>",
            content_raw=question, meta={}, parent_id=effective_parent,
        )
        result = _run_rag(question, top_k=top_k, level=level)
        insert_message(
            conn, chat_id, None, "advisor",
            content_html=result["answer"], content_raw=None,
            meta={"t_ms": result["t_ms"], "sources": result["sources"]},
            parent_id=user_msg_id,
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "titles": result["titles"],
            "t_ms": result["t_ms"],
            "chat_id": chat_id,
            "user_msg_id": user_msg_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ========== /review (PDF/TXT/DOCX) + persistence + branching ==========
@app.post("/review")
def review_endpoint(
    authorization: Optional[str] = Header(None),
    chat_id: Optional[str] = Form(None),
    parent_id: Optional[str] = Form(None),
    question: str = Form(""),
    files: List[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user),
):
    require_auth(authorization)
    check_rate_limit()
    conn = db()
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        chat_id = ensure_chat(conn, user_id, chat_id)
        effective_parent = parent_id or _latest_leaf(conn, chat_id)
        user_msg_id = insert_message(
            conn, chat_id, user_id, "user",
            content_html=f"<p>{question or '(Uploaded documents for review.)'}</p>",
            content_raw=question, meta={"upload": True},
            parent_id=effective_parent,
        )

        texts: List[str] = []
        for f in files:
            name = (f.filename or "").lower()
            raw  = f.file.read(UPLOAD_MAX_BYTES + 1)
            if len(raw) > UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413,
                                    detail=f"{f.filename} exceeds {UPLOAD_MAX_BYTES//1024//1024}MB limit.")
            if name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    pages = []
                    for p in reader.pages:
                        try:    pages.append(p.extract_text() or "")
                        except Exception: pages.append("")
                    texts.append("\n".join(pages))
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse PDF: {f.filename} ({e})")
            elif name.endswith(".txt"):
                try:    texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception: texts.append(raw.decode("latin-1", errors="ignore"))
            elif name.endswith(".docx"):
                try:
                    try:
                        import docx  # python-docx
                        doc = docx.Document(io.BytesIO(raw))
                        paras = [p.text for p in doc.paragraphs if p.text]
                        texts.append("\n".join(paras))
                    except Exception:
                        with zipfile.ZipFile(io.BytesIO(raw)) as z:
                            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                            stripped = re.sub(r"<[^>]+>", " ", xml)
                            stripped = re.sub(r"\s+", " ", stripped).strip()
                            texts.append(stripped)
                except Exception as e:
                    raise HTTPException(status_code=415, detail=f"Failed to parse DOCX: {f.filename} ({e})")
            else:
                raise HTTPException(status_code=415,
                                    detail=f"Unsupported file type: {f.filename} (only PDF/TXT/DOCX)")

        merged = "\n---\n".join([t for t in texts if t.strip()])
        chunks  = [merged[i:i + 2000] for i in range(0, len(merged), 2000)][:MAX_SNIPPETS]
        pseudo  = [{"title": "Uploaded Document", "level": "L5", "page": "?",
                    "version": "", "score": 1.0, "meta": {}}]
        html    = synthesize_html(question or "Please analyze the attached materials.", pseudo, chunks)

        insert_message(
            conn, chat_id, None, "advisor",
            content_html=html, content_raw=None, meta={"upload": True},
            parent_id=user_msg_id,
        )
        return {"answer": html, "t_ms": 0, "chat_id": chat_id, "user_msg_id": user_msg_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# ==========================================================
# /draft — generate full HTML fiduciary resolution (internal RAG)
# ==========================================================
@app.post("/draft")
async def draft(request: Request,
                authorization: Optional[str] = Header(None)):
    """
    Produces a comprehensive, HTML-formatted trustee resolution using:
      (1) the internal RAG pipeline for citations, and
      (2) an OpenAI completion with the fiduciary system prompt.
    """
    require_auth(authorization)
    check_rate_limit()

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or missing JSON body.")

    question = (data.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field.")

    # 1. Internal RAG — no self-HTTP call
    rag = _run_rag(question, top_k=12)
    sources = rag.get("sources", [])
    src_text = "\n".join([
        f"{s.get('title','')} (L{s.get('level','')}, p.{s.get('page','')})" for s in sources
    ])

    # 2. Compose prompt
    system_prompt = _effective_system_prompt()
    model = os.environ.get("SYNTH_MODEL", SYNTH_MODEL)
    max_tokens = int(os.environ.get("MAX_OUT_TOKENS", str(MAX_OUT_TOKENS)))

    draft_instructions = f"""{system_prompt}

Use the following citations where applicable:
{src_text}

The trustee request is:
{question}

Produce a comprehensive, legally formatted trustee resolution in rich HTML.

Requirements:
- Write as if by an experienced fiduciary attorney preparing an official trust record.
- Use clear HTML structure: <h1> title, <h2> section headings, <p> narrative paragraphs, and <ul>/<li> lists for details.
- Sections, in order:
   1. Title of the resolution
   2. Date and Location
   3. Trust Details -- name, type, situs/jurisdiction, trustee(s)
   4. Recitals (WHEREAS clauses)
   5. Resolution Section -- detailed authorization
   6. Legal Basis and References -- integrate citations from the sources above
   7. Execution Section -- closing paragraph and signature block
- Only include sections for which data is available.
- Do not insert underscores or placeholders for missing data.
- Output must be valid, styled HTML ready for web display.
""".strip()

    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": draft_instructions},
            ],
        )
        answer = res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "sources": sources, "model": model}

# ==========================================================
# /chat — conversational fiduciary advisor (internal RAG)
# ==========================================================
@app.post("/chat")
async def chat_endpoint(request: Request,
                        authorization: Optional[str] = Header(None)):
    require_auth(authorization)
    check_rate_limit()

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or missing JSON body.")

    message = (data.get("message") or "").strip()
    thread_id = data.get("thread_id", "default")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' field.")

    rag = _run_rag(message, top_k=12)
    sources = rag.get("sources", [])
    src_text = "\n".join([
        f"{s.get('title','')} (L{s.get('level','')}, p.{s.get('page','')})" for s in sources
    ])

    system_prompt = _effective_system_prompt()
    model = os.environ.get("SYNTH_MODEL", SYNTH_MODEL)
    max_tokens = int(os.environ.get("MAX_OUT_TOKENS", str(MAX_OUT_TOKENS)))

    composite_prompt = f"""{system_prompt}

Use these citations where relevant:
{src_text}

Question:
{message}

Respond as the Private Fiduciary Advisor: formal, clear, authoritative, but conversational.
Include citations inline where appropriate (e.g., IRC §642(c)(2); Treas. Reg. §1.642(c)-2).
Answer in full sentences, not bullet points unless listing authorities.
""".strip()

    try:
        res = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": composite_prompt},
            ],
        )
        answer = res.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return {"answer": answer, "sources": sources, "thread_id": thread_id, "model": model}
