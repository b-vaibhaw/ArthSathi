"""
rag_engine.py
==============
ArthaSathi — Retrieval Augmented Generation + User Memory
- FAISSVectorStore   : semantic search over financial knowledge
- KnowledgeBaseBuilder : loads RBI rules, GST, tax, negotiation scripts
- UserMemory         : per-user SQLite storage (debts, transactions, history)
- ContextBuilder     : assembles LLM prompt context from RAG + memory
"""

import os, json, sqlite3, hashlib, time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date


# ─────────────────────────────────────────────────────────────
# 1. FAISS VECTOR STORE
# ─────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Semantic search over financial knowledge base.
    Uses multilingual MiniLM — works for all 9 Indian languages.
    Falls back to keyword search if FAISS / SentenceTransformers unavailable.
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DIM        = 384

    def __init__(self, index_path: Optional[str] = None):
        self.encoder   = None
        self.index     = None
        self.documents: List[Dict] = []
        self._load_encoder()
        if index_path and Path(index_path).exists():
            self.load(index_path)

    def _load_encoder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.MODEL_NAME)
        except ImportError:
            print("[RAG] sentence-transformers not found. pip install sentence-transformers")

    def _embed(self, text: str):
        import numpy as np
        if self.encoder:
            v = self.encoder.encode([text])[0].astype("float32")
        else:
            # Deterministic fallback (NOT for production)
            rng = __import__("random").Random(int(hashlib.md5(text.encode()).hexdigest(), 16))
            v   = np.array([rng.gauss(0,1) for _ in range(self.DIM)], dtype="float32")
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    def _ensure_index(self):
        if self.index is None:
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.DIM)
            except ImportError:
                pass  # keyword fallback

    def add(self, text: str, metadata: Dict = None):
        self._ensure_index()
        if self.index is not None:
            import numpy as np
            self.index.add(self._embed(text).reshape(1, -1))
        self.documents.append({"text": text, "metadata": metadata or {}})

    def add_batch(self, texts: List[str], metadatas: List[Dict] = None):
        self._ensure_index()
        if metadatas is None:
            metadatas = [{} for _ in texts]
        import numpy as np
        embs = np.stack([self._embed(t) for t in texts])
        if self.index is not None:
            self.index.add(embs)
        for t, m in zip(texts, metadatas):
            self.documents.append({"text": t, "metadata": m or {}})

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if not self.documents:
            return []
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return self._keyword_search(query, k)
        import numpy as np
        q = self._embed(query).reshape(1, -1)
        scores, idxs = self.index.search(q, min(k, len(self.documents)))
        results = []
        for sc, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.documents):
                d = dict(self.documents[i]); d["score"] = float(sc)
                results.append(d)
        return results

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        qw = set(query.lower().split())
        scored = sorted(
            [(len(qw & set(d["text"].lower().split())), d) for d in self.documents],
            key=lambda x: x[0], reverse=True
        )
        return [d for sc, d in scored[:k] if sc > 0]

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(Path(path) / "faiss.index"))
        with open(Path(path) / "docs.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)
        print(f"[RAG] Saved {len(self.documents)} docs → {path}/")

    def load(self, path: str):
        try:
            import faiss
            self.index = faiss.read_index(str(Path(path) / "faiss.index"))
        except Exception as e:
            print(f"[RAG] FAISS index load failed: {e}")
        p = Path(path) / "docs.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                self.documents = json.load(f)
        print(f"[RAG] Loaded {len(self.documents)} docs from {path}/")


# ─────────────────────────────────────────────────────────────
# 2. KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────

KNOWLEDGE_DOCS = [
    # RBI rules
    ("RBI Fair Practices Code: Banks cannot contact borrowers before 8AM or after 8PM. "
     "Recovery agents must show ID. File complaint at RBI Ombudsman — free, no lawyer needed.",
     {"source":"RBI","category":"borrower_rights","lang":"en"}),

    ("बैंक के नियम: बैंक आपको रात 8 बजे के बाद या सुबह 8 बजे से पहले फोन नहीं कर सकता। "
     "शिकायत के लिए RBI Ombudsman पूरी तरह मुफ्त है। कोई वकील नहीं चाहिए।",
     {"source":"RBI","category":"borrower_rights","lang":"hi"}),

    ("RBI Digital Lending Guidelines 2022: Digital lenders must show real APR upfront. "
     "Cannot auto-debit without explicit consent. Cooling-off period: 3 days to cancel loan.",
     {"source":"RBI","category":"digital_lending","lang":"en"}),

    ("NBFC-MFI interest rate cap: RBI has capped microfinance lending at maximum 24% per year "
     "for group loans (JLG). Any higher rate violates RBI regulations — file complaint immediately.",
     {"source":"RBI","category":"microfinance","lang":"en"}),

    # Credit card
    ("Credit card interest: India credit cards charge 36-42% per year. "
     "Paying only minimum payment on Rs 10,000 debt takes 7+ years to clear. Always pay full bill.",
     {"source":"knowledge","category":"credit_card","lang":"en"}),

    ("क्रेडिट कार्ड ट्रैप: सिर्फ minimum payment करने पर 10,000 का कर्ज़ चुकाने में 7+ साल लग सकते हैं। "
     "हमेशा पूरा बिल भरें। नहीं कर सकते? तो bank से EMI conversion के लिए बोलें।",
     {"source":"knowledge","category":"credit_card","lang":"hi"}),

    # Moneylender
    ("Moneylender usury: If a moneylender charges more than 2% per month (24% per year), "
     "it is illegal under the State Moneylenders Act in most Indian states. "
     "File complaint with District Collector or police.",
     {"source":"legal","category":"moneylender","lang":"en"}),

    ("साहूकार कानून: अगर साहूकार 2% प्रति महीने (24% साल) से ज़्यादा ब्याज ले रहा है, "
     "तो यह ज़्यादातर राज्यों में गैरकानूनी है। District Collector या Police में शिकायत करें।",
     {"source":"legal","category":"moneylender","lang":"hi"}),

    # GST
    ("GST threshold: Annual turnover below Rs 40 lakh (goods) or Rs 20 lakh (services) "
     "does NOT need GST registration. Special category states threshold: Rs 10 lakh.",
     {"source":"GST","category":"gst_registration","lang":"en"}),

    ("GST Composition Scheme: Turnover up to Rs 1.5 crore — pay just 1% GST (traders), "
     "5% (restaurants). Simple quarterly filing. Best for small shops and vendors.",
     {"source":"GST","category":"gst_composition","lang":"en"}),

    ("GST नियम: 40 लाख से कम सालाना कारोबार (माल) या 20 लाख से कम (सेवाएं) — "
     "GST रजिस्ट्रेशन जरूरी नहीं। Composition Scheme: सिर्फ 1% GST, quarterly filing।",
     {"source":"GST","category":"gst_registration","lang":"hi"}),

    ("GST Composition Scheme: वार्षिक उलाढाल 1.5 कोटींपर्यंत — फक्त 1% GST (व्यापाऱ्यांसाठी), "
     "5% (रेस्टॉरंटसाठी). तिमाही filing. छोट्या दुकानदारांसाठी सर्वोत्तम.",
     {"source":"GST","category":"gst_composition","lang":"mr"}),

    ("GST விதிமுறைகள்: ஆண்டு விற்பனை ரூ.40 லட்சத்திற்கும் குறைவாக இருந்தால் "
     "GST பதிவு தேவையில்லை. Composition Scheme: வெறும் 1% GST.",
     {"source":"GST","category":"gst_registration","lang":"ta"}),

    # Income tax
    ("Income Tax: Under new regime, income up to Rs 7 lakh has ZERO tax (Section 87A rebate). "
     "Gig workers, street vendors, and informal workers earning under 7L pay no tax. "
     "Still file ITR — it builds credit history.",
     {"source":"IncomeTax","category":"tax_rebate","lang":"en"}),

    ("Income Tax: ITR-4 for small business — declare just 8% of turnover as income "
     "(6% for digital receipts). No books, no audit needed up to Rs 2 crore turnover.",
     {"source":"IncomeTax","category":"itr4","lang":"en"}),

    ("आयकर: नए टैक्स regime में 7 लाख तक की आमदनी पर शून्य Tax। Section 87A की छूट। "
     "ITR-4 से छोटे कारोबारी सिर्फ turnover का 8% income दिखाएं — no books, no audit।",
     {"source":"IncomeTax","category":"tax_rebate","lang":"hi"}),

    ("வருமான வரி: புதிய வரி முறையில் ரூ.7 லட்சம் வரை வருமானத்திற்கு வரி இல்லை. "
     "சிறு தொழில் செய்பவர்கள் ITR-4 மூலம் turnover-ன் 8% மட்டும் வருமானமாக காட்டலாம்.",
     {"source":"IncomeTax","category":"tax_rebate","lang":"ta"}),

    ("ಆದಾಯ ತೆರಿಗೆ: ಹೊಸ ತೆರಿಗೆ ಪದ್ಧತಿಯಲ್ಲಿ ₹7 ಲಕ್ಷದವರೆಗೆ ಯಾವುದೇ ತೆರಿಗೆ ಇಲ್ಲ। "
     "ಸಣ್ಣ ವ್ಯಾಪಾರಿಗಳು ITR-4 ಮೂಲಕ turnover-ನ 8% ಮಾತ್ರ ಆದಾಯ ತೋರಿಸಬಹುದು.",
     {"source":"IncomeTax","category":"tax_rebate","lang":"kn"}),

    ("আয়কর: নতুন কর ব্যবস্থায় ৭ লাখ পর্যন্ত আয়ে কোনো কর নেই। "
     "ছোট ব্যবসায়ীরা ITR-4 দিয়ে শুধু টার্নওভারের ৮% আয় দেখাতে পারবেন।",
     {"source":"IncomeTax","category":"tax_rebate","lang":"bn"}),

    ("ఆదాయ పన్ను: కొత్త పన్ను విధానంలో ₹7 లక్షల వరకు ఆదాయానికి పన్ను లేదు. "
     "చిన్న వ్యాపారులు ITR-4 ద్వారా టర్నోవర్‌లో 8% మాత్రమే ఆదాయంగా చూపించవచ్చు.",
     {"source":"IncomeTax","category":"tax_rebate","lang":"te"}),

    # MUDRA
    ("MUDRA Loan: Government scheme — no collateral. "
     "Shishu: up to Rs 50,000 (interest ~10%). Kishor: up to Rs 5 lakh. "
     "Apply at any PSU bank, NBFC. mudra.org.in or nearest bank.",
     {"source":"MUDRA","category":"govt_loan","lang":"en"}),

    ("MUDRA लोन: सरकारी योजना, कोई गारंटी नहीं। "
     "शिशु: 50,000 तक (~10% ब्याज)। किशोर: 5 लाख तक। "
     "किसी भी सरकारी बैंक या NBFC में apply करें।",
     {"source":"MUDRA","category":"govt_loan","lang":"hi"}),

    # Debt strategies
    ("Debt Avalanche: Pay minimum on all loans. Put all extra money on HIGHEST interest rate loan first. "
     "Mathematically optimal — saves maximum total interest paid.",
     {"source":"strategy","category":"debt_payoff","lang":"en"}),

    ("Debt Snowball: Pay minimum on all loans. Put extra money on SMALLEST balance first. "
     "Psychologically better — quick wins keep you motivated.",
     {"source":"strategy","category":"debt_payoff","lang":"en"}),

    ("ऋण चुकाने की रणनीति: सबसे ज़्यादा ब्याज वाला लोन पहले चुकाएं (Avalanche method) — "
     "यह सबसे कम कुल ब्याज देगा। या सबसे छोटा लोन पहले (Snowball) — मनोबल बढ़ेगा।",
     {"source":"strategy","category":"debt_payoff","lang":"hi"}),

    # Negotiation
    ("Credit card settlement script: 'I am facing financial hardship. I want to request "
     "a one-time settlement at 60% of outstanding. Please transfer me to settlements team.' "
     "Banks have hidden settlements desks. Persistence works.",
     {"source":"template","category":"negotiation","lang":"en"}),

    ("EMI restructuring script: 'Due to income reduction, I cannot pay current EMI. "
     "I request restructuring to lower EMI for 12 months. I can provide income documents.' "
     "RBI guidelines require banks to consider genuine hardship cases.",
     {"source":"template","category":"negotiation","lang":"en"}),

    ("कर्ज़ settlement के लिए बात: 'मैं आर्थिक तंगी में हूं। बकाया राशि का 60% एकमुश्त settlement "
     "चाहता हूं। कृपया settlements department से जोड़ें।' — बैंकों का hidden settlement desk होता है।",
     {"source":"template","category":"negotiation","lang":"hi"}),

    # Consumer rights
    ("Consumer Protection Act 2019: File complaint at consumer forum against unfair bank practices. "
     "No lawyer needed. Rs 200 fee for claims up to Rs 1 crore. Online: consumerhelpline.gov.in",
     {"source":"legal","category":"consumer_rights","lang":"en"}),

    ("उपभोक्ता अधिकार: बैंक के खिलाफ Consumer Forum में मुफ्त शिकायत। "
     "1 करोड़ तक के दावों के लिए सिर्फ ₹200 fee। ऑनलाइन: consumerhelpline.gov.in",
     {"source":"legal","category":"consumer_rights","lang":"hi"}),

    # Bhojpuri
    ("कर्ज के बारे में: रउआ के सबसे ज़्यादा ब्याज वाला कर्ज पहिले चुकाए के चाहीं। "
     "बैंक से settlement के लिए बात कर सकत बानी। RBI Ombudsman में मुफ्त शिकायत होला।",
     {"source":"knowledge","category":"debt_strategy","lang":"bho"}),

    # Assamese
    ("ঋণ পৰিশোধৰ কৌশল: সৰ্বাধিক সুদৰ হাৰৰ ঋণ আগতে পৰিশোধ কৰক (Avalanche পদ্ধতি)। "
     "বেংকৰ সৈতে EMI পুনৰ গঠনৰ বাবে কথা পাতক। RBI Ombudsman-ত বিনামূলীয়া অভিযোগ।",
     {"source":"knowledge","category":"debt_strategy","lang":"as"}),
]


def build_knowledge_base(store: FAISSVectorStore, extra_dir: Optional[str] = None):
    """Load all financial knowledge into vector store"""
    texts = [d[0] for d in KNOWLEDGE_DOCS]
    metas = [d[1] for d in KNOWLEDGE_DOCS]
    store.add_batch(texts, metas)
    if extra_dir and Path(extra_dir).exists():
        n = 0
        for f in Path(extra_dir).glob("*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        store.add(obj.get("text",""), {"source": f.stem})
                        n += 1
                    except Exception:
                        pass
        print(f"[RAG] Loaded {n} extra docs from {extra_dir}")
    print(f"[RAG] Knowledge base ready: {len(store.documents)} documents")


# ─────────────────────────────────────────────────────────────
# 3. USER MEMORY (SQLite)
# ─────────────────────────────────────────────────────────────

class UserMemory:
    """
    Per-user persistent memory stored in SQLite.
    Stores: financial profile, debts, transactions, conversation history.
    Thread-safe with check_same_thread=False for async usage.
    """

    def __init__(self, db_path: str = "arthasathi_users.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id        TEXT PRIMARY KEY,
                language       TEXT DEFAULT 'hi',
                monthly_income REAL DEFAULT 0.0,
                literacy_level TEXT DEFAULT 'basic',
                channel        TEXT DEFAULT 'whatsapp',
                created_at     TEXT DEFAULT (datetime('now')),
                updated_at     TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS debts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                name         TEXT,
                principal    REAL,
                annual_rate  REAL,
                min_payment  REAL,
                lender_type  TEXT,
                added_at     TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                type         TEXT,
                amount       REAL,
                category     TEXT,
                description  TEXT,
                date_str     TEXT,
                lang         TEXT,
                added_at     TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                role         TEXT,
                content      TEXT,
                lang         TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS kv_store (
                user_id      TEXT NOT NULL,
                key          TEXT NOT NULL,
                value        TEXT,
                updated_at   TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, key)
            );
        """)
        self.db.commit()

    # ── Users ─────────────────────────────────────────────────

    def get_or_create_user(self, user_id: str, language: str = "hi") -> Dict:
        row = self.db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            self.db.execute(
                "INSERT INTO users (user_id, language) VALUES (?, ?)",
                (user_id, language)
            )
            self.db.commit()
            return {"user_id": user_id, "language": language,
                    "monthly_income": 0.0, "is_new": True}
        return dict(row)

    def update_user(self, user_id: str, **kwargs):
        allowed = {"language", "monthly_income", "literacy_level", "channel"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        sets   = ", ".join(f"{k}=?" for k in updates)
        values = list(updates.values()) + [user_id]
        self.db.execute(
            f"UPDATE users SET {sets}, updated_at=datetime('now') WHERE user_id=?", values
        )
        self.db.commit()

    # ── Debts ─────────────────────────────────────────────────

    def add_debt(self, user_id: str, name: str, principal: float,
                 annual_rate: float, min_payment: float, lender_type: str = "bank"):
        self.db.execute(
            "INSERT INTO debts (user_id,name,principal,annual_rate,min_payment,lender_type) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, name, principal, annual_rate, min_payment, lender_type)
        )
        self.db.commit()

    def get_debts(self, user_id: str) -> List[Dict]:
        rows = self.db.execute(
            "SELECT * FROM debts WHERE user_id = ? ORDER BY annual_rate DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_debt(self, user_id: str, debt_id: int):
        self.db.execute("DELETE FROM debts WHERE id=? AND user_id=?", (debt_id, user_id))
        self.db.commit()

    # ── Transactions ──────────────────────────────────────────

    def add_transaction(self, user_id: str, txn_type: str, amount: float,
                        category: str = "other", description: str = "",
                        date_str: str = "", lang: str = "hi"):
        if not date_str:
            date_str = str(date.today())
        self.db.execute(
            "INSERT INTO transactions (user_id,type,amount,category,description,date_str,lang) "
            "VALUES (?,?,?,?,?,?,?)",
            (user_id, txn_type, amount, category, description, date_str, lang)
        )
        self.db.commit()

    def get_transactions(self, user_id: str, days: int = 90,
                         txn_type: Optional[str] = None) -> List[Dict]:
        query  = "SELECT * FROM transactions WHERE user_id=? AND date_str >= date('now', ?)"
        params = [user_id, f"-{days} days"]
        if txn_type:
            query  += " AND type=?"
            params += [txn_type]
        query += " ORDER BY date_str DESC"
        rows = self.db.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_monthly_summary(self, user_id: str, months: int = 3) -> Dict:
        txns = self.get_transactions(user_id, days=months * 30)
        income  = sum(t["amount"] for t in txns if t["type"] == "income")
        expense = sum(t["amount"] for t in txns if t["type"] == "expense")
        n_months = max(months, 1)
        return {
            "total_income":    round(income, 2),
            "total_expense":   round(expense, 2),
            "net":             round(income - expense, 2),
            "avg_monthly_income":  round(income / n_months, 2),
            "avg_monthly_expense": round(expense / n_months, 2),
            "n_transactions":  len(txns),
        }

    # ── Conversation History ──────────────────────────────────

    def add_message(self, user_id: str, role: str, content: str, lang: str = "hi"):
        self.db.execute(
            "INSERT INTO messages (user_id,role,content,lang) VALUES (?,?,?,?)",
            (user_id, role, content, lang)
        )
        self.db.commit()

    def get_history(self, user_id: str, last_n: int = 10) -> List[Dict]:
        rows = self.db.execute(
            "SELECT role, content, lang FROM messages WHERE user_id=? "
            "ORDER BY id DESC LIMIT ?",
            (user_id, last_n)
        ).fetchall()
        return list(reversed([dict(r) for r in rows]))

    # ── Key-Value store ───────────────────────────────────────

    def set(self, user_id: str, key: str, value):
        self.db.execute(
            "INSERT OR REPLACE INTO kv_store (user_id,key,value,updated_at) "
            "VALUES (?,?,?,datetime('now'))",
            (user_id, key, json.dumps(value, ensure_ascii=False))
        )
        self.db.commit()

    def get(self, user_id: str, key: str, default=None):
        row = self.db.execute(
            "SELECT value FROM kv_store WHERE user_id=? AND key=?", (user_id, key)
        ).fetchone()
        return json.loads(row[0]) if row else default

    # ── Full profile ──────────────────────────────────────────

    def get_full_profile(self, user_id: str) -> Dict:
        user  = self.get_or_create_user(user_id)
        debts = self.get_debts(user_id)
        summary = self.get_monthly_summary(user_id)
        history = self.get_history(user_id, last_n=6)
        total_debt     = sum(d["principal"] for d in debts)
        total_min_emi  = sum(d["min_payment"] for d in debts)
        monthly_income = user.get("monthly_income") or summary["avg_monthly_income"]
        return {
            "user_id":       user_id,
            "language":      user.get("language", "hi"),
            "monthly_income": round(monthly_income, 2),
            "debts":         debts,
            "total_debt":    round(total_debt, 2),
            "total_emi":     round(total_min_emi, 2),
            "txn_summary":   summary,
            "recent_history": history,
            "debt_to_income": round(total_min_emi / max(monthly_income, 1) * 100, 1),
        }


# ─────────────────────────────────────────────────────────────
# 4. CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Builds the complete LLM prompt context by combining:
    1. Relevant knowledge from FAISS vector store
    2. User's financial profile from SQLite memory
    3. Recent conversation history
    4. Financial engine results (computed numbers)
    """

    MAX_KNOWLEDGE_CHARS = 800
    MAX_PROFILE_CHARS   = 500
    MAX_HISTORY_TURNS   = 5

    def __init__(self, vector_store: FAISSVectorStore, user_memory: UserMemory):
        self.store  = vector_store
        self.memory = user_memory

    def build(self, user_id: str, user_message: str,
              engine_result: Optional[Dict] = None,
              language: str = "hi") -> str:
        """
        Assemble full context string for LLM prompt injection.
        Returns a structured context block appended before the user message.
        """
        parts = []

        # ── 1. Relevant knowledge ─────────────────────────────
        docs = self.store.search(user_message, k=3)
        if docs:
            knowledge = "\n".join(
                f"- {d['text'][:200]}" for d in docs
            )[:self.MAX_KNOWLEDGE_CHARS]
            parts.append(f"[RELEVANT KNOWLEDGE]\n{knowledge}")

        # ── 2. User profile ───────────────────────────────────
        profile = self.memory.get_full_profile(user_id)
        profile_text = self._format_profile(profile, language)
        if profile_text:
            parts.append(f"[USER PROFILE]\n{profile_text}")

        # ── 3. Engine computation result ──────────────────────
        if engine_result:
            eng_text = json.dumps(engine_result, ensure_ascii=False)[:400]
            parts.append(f"[CALCULATION RESULT]\n{eng_text}")

        # ── 4. Conversation history ───────────────────────────
        history = profile.get("recent_history", [])[-self.MAX_HISTORY_TURNS:]
        if history:
            hist_text = "\n".join(
                f"{m['role'].upper()}: {m['content'][:100]}" for m in history
            )
            parts.append(f"[RECENT CONVERSATION]\n{hist_text}")

        return "\n\n".join(parts)

    def _format_profile(self, profile: Dict, language: str) -> str:
        lines = []
        income = profile.get("monthly_income", 0)
        if income > 0:
            lines.append(f"Monthly income: Rs {income:,.0f}")

        debts = profile.get("debts", [])
        if debts:
            lines.append(f"Active debts ({len(debts)}):")
            for d in debts[:3]:
                lines.append(
                    f"  - {d['name']}: Rs {d['principal']:,.0f} @ {d['annual_rate']}%"
                )
            dti = profile.get("debt_to_income", 0)
            lines.append(f"Debt-to-income ratio: {dti}%")

        summary = profile.get("txn_summary", {})
        if summary.get("n_transactions", 0) > 0:
            lines.append(
                f"Monthly avg: income Rs {summary['avg_monthly_income']:,.0f}, "
                f"expense Rs {summary['avg_monthly_expense']:,.0f}"
            )

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 5. MASTER RAG ENGINE
# ─────────────────────────────────────────────────────────────

class ArthaSathiRAG:
    """
    Top-level RAG engine — single entry point for the API layer.
    Combines vector store + user memory + context building.
    """

    def __init__(self, index_path: str = "rag_index",
                 db_path: str = "arthasathi_users.db"):
        self.store   = FAISSVectorStore(index_path=index_path if Path(index_path).exists() else None)
        self.memory  = UserMemory(db_path)
        self.context = ContextBuilder(self.store, self.memory)

        # Build knowledge base if store is empty
        if len(self.store.documents) == 0:
            print("[RAG] Building knowledge base from scratch...")
            build_knowledge_base(self.store)
            self.store.save(index_path)

    def get_context(self, user_id: str, message: str,
                    engine_result: Optional[Dict] = None,
                    language: str = "hi") -> str:
        return self.context.build(user_id, message, engine_result, language)

    def save_user_message(self, user_id: str, content: str, lang: str = "hi"):
        self.memory.add_message(user_id, "user", content, lang)

    def save_assistant_message(self, user_id: str, content: str, lang: str = "hi"):
        self.memory.add_message(user_id, "assistant", content, lang)

    def save_transaction(self, user_id: str, txn: Dict):
        if txn and txn.get("amount", 0) > 0:
            self.memory.add_transaction(
                user_id,
                txn_type=txn.get("type", "expense"),
                amount=txn["amount"],
                category=txn.get("category", "other"),
                description=txn.get("description", ""),
                lang=txn.get("lang", "hi"),
            )

    def add_debt(self, user_id: str, **kwargs):
        self.memory.add_debt(user_id, **kwargs)

    def get_profile(self, user_id: str) -> Dict:
        return self.memory.get_full_profile(user_id)

    def detect_language(self, text: str) -> str:
        """Fast language detection using Unicode ranges"""
        counts: Dict[str, int] = {}
        for ch in text:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F: counts["hi"] = counts.get("hi", 0) + 1   # Devanagari (hi/mr/bho)
            elif 0x0B80 <= cp <= 0x0BFF: counts["ta"] = counts.get("ta", 0) + 1  # Tamil
            elif 0x0C80 <= cp <= 0x0CFF: counts["kn"] = counts.get("kn", 0) + 1  # Kannada
            elif 0x0980 <= cp <= 0x09FF: counts["bn"] = counts.get("bn", 0) + 1  # Bengali/Assamese
            elif 0x0C00 <= cp <= 0x0C7F: counts["te"] = counts.get("te", 0) + 1  # Telugu
            elif ch.isascii() and ch.isalpha(): counts["en"] = counts.get("en", 0) + 1
        if not counts:
            return "hi"  # default
        detected = max(counts, key=counts.get)
        # Disambiguation: Assamese and Bengali share Unicode block
        if detected == "bn":
            assamese_chars = ["ৰ", "ল", "ক্ষ"]
            if any(c in text for c in assamese_chars):
                return "as"
        return detected


# ─────────────────────────────────────────────────────────────
# 6. QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("ArthaSathi RAG Engine — Test")
    print("=" * 55)

    rag = ArthaSathiRAG(index_path="/tmp/test_rag_index",
                        db_path="/tmp/test_users.db")

    # Test knowledge search
    queries = [
        "mera credit card ka 30000 debt hai kya karu",
        "GST registration ki zarurat hai kya",
        "bank wale raat ko phone kar rahe hain",
        "income tax kitna bharna padega",
    ]
    print("\n── Knowledge Search Tests ──")
    for q in queries:
        results = rag.store.search(q, k=2)
        print(f"\nQ: {q[:50]}")
        for r in results:
            print(f"  → [{r['metadata'].get('category','')}] {r['text'][:80]}...")

    # Test user memory
    uid = "test_user_91"
    rag.memory.get_or_create_user(uid, "hi")
    rag.memory.update_user(uid, monthly_income=15000)
    rag.add_debt(uid, name="HDFC CC", principal=30000,
                 annual_rate=36, min_payment=900, lender_type="bank")
    rag.memory.add_transaction(uid, "income", 15000, "salary", "Monthly salary")
    rag.memory.add_transaction(uid, "expense", 3000, "rent", "Room rent")

    profile = rag.get_profile(uid)
    print(f"\n── User Profile ──")
    print(f"  Income: Rs {profile['monthly_income']:,}")
    print(f"  Debts:  {len(profile['debts'])} | Total: Rs {profile['total_debt']:,}")
    print(f"  DTI:    {profile['debt_to_income']}%")

    # Test context building
    ctx = rag.get_context(uid, "mera loan kaise bharu", language="hi")
    print(f"\n── Built Context (first 300 chars) ──")
    print(ctx[:300])

    # Test language detection
    tests = [
        ("mera loan hai", "hi"),
        ("my debt is too high", "en"),
        ("माझ्या कर्जावर खूप व्याज", "hi"),
        ("என் கடன் அதிகமாக உள்ளது", "ta"),
        ("ನನ್ನ ಸಾಲ ತೀರಿಸಲು ಸಾಧ್ಯವಾಗುತ್ತಿಲ್ಲ", "kn"),
    ]
    print("\n── Language Detection ──")
    for text, expected in tests:
        detected = rag.detect_language(text)
        ok = "✓" if detected == expected else "✗"
        print(f"  {ok} '{text[:35]}' → {detected} (expected {expected})")

    print("\nRAG engine tests complete!")
