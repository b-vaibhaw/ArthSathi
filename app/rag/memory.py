"""
memory.py
=========
Per-user persistent memory in SQLite: profile, debts, transactions,
conversation history. This is what lets the agent remember a user across
messages without re-asking their income/debts every time.
"""

import json
import sqlite3
from datetime import date
from typing import Dict, List, Optional


class UserMemory:
    def __init__(self, db_path: str = "arthasathi.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id        TEXT PRIMARY KEY,
                language       TEXT DEFAULT 'en',
                monthly_income REAL DEFAULT 0.0,
                channel        TEXT DEFAULT 'web',
                created_at     TEXT DEFAULT (datetime('now')),
                updated_at     TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS debts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                name         TEXT, principal REAL, annual_rate REAL,
                min_payment  REAL, lender_type TEXT,
                added_at     TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                type         TEXT, amount REAL, category TEXT,
                description  TEXT, date_str TEXT, lang TEXT,
                added_at     TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                role         TEXT, content TEXT, lang TEXT,
                created_at   TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS kv_store (
                user_id      TEXT NOT NULL, key TEXT NOT NULL, value TEXT,
                updated_at   TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, key)
            );
        """)
        self.db.commit()

    # ── Users ────────────────────────────────────────────────
    def get_or_create_user(self, user_id: str, language: str = "en") -> Dict:
        row = self.db.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            self.db.execute("INSERT INTO users (user_id, language) VALUES (?,?)", (user_id, language))
            self.db.commit()
            return {"user_id": user_id, "language": language, "monthly_income": 0.0, "is_new": True}
        return dict(row)

    def update_user(self, user_id: str, **kwargs):
        self.get_or_create_user(user_id)
        allowed = {"language", "monthly_income", "channel"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        sets = ", ".join(f"{k}=?" for k in updates)
        self.db.execute(f"UPDATE users SET {sets}, updated_at=datetime('now') WHERE user_id=?",
                         list(updates.values()) + [user_id])
        self.db.commit()

    # ── Debts ────────────────────────────────────────────────
    def add_debt(self, user_id: str, name: str, principal: float, annual_rate: float,
                 min_payment: float, lender_type: str = "bank"):
        self.get_or_create_user(user_id)
        self.db.execute(
            "INSERT INTO debts (user_id,name,principal,annual_rate,min_payment,lender_type) VALUES (?,?,?,?,?,?)",
            (user_id, name, principal, annual_rate, min_payment, lender_type))
        self.db.commit()

    def get_debts(self, user_id: str) -> List[Dict]:
        rows = self.db.execute("SELECT * FROM debts WHERE user_id=? ORDER BY annual_rate DESC", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def clear_debts(self, user_id: str):
        self.db.execute("DELETE FROM debts WHERE user_id=?", (user_id,))
        self.db.commit()

    # ── Transactions ─────────────────────────────────────────
    def add_transaction(self, user_id: str, txn_type: str, amount: float,
                         category: str = "other", description: str = "",
                         date_str: str = "", lang: str = "en"):
        self.get_or_create_user(user_id)
        self.db.execute(
            "INSERT INTO transactions (user_id,type,amount,category,description,date_str,lang) VALUES (?,?,?,?,?,?,?)",
            (user_id, txn_type, amount, category, description, date_str or str(date.today()), lang))
        self.db.commit()

    def get_transactions(self, user_id: str, days: int = 90) -> List[Dict]:
        rows = self.db.execute(
            "SELECT * FROM transactions WHERE user_id=? AND date_str >= date('now', ?) ORDER BY date_str DESC",
            (user_id, f"-{days} days")).fetchall()
        return [dict(r) for r in rows]

    def get_monthly_summary(self, user_id: str, months: int = 3) -> Dict:
        txns = self.get_transactions(user_id, days=months * 30)
        income = sum(t["amount"] for t in txns if t["type"] == "income")
        expense = sum(t["amount"] for t in txns if t["type"] == "expense")
        n_months = max(months, 1)
        return {
            "total_income": round(income, 2), "total_expense": round(expense, 2),
            "net": round(income - expense, 2),
            "avg_monthly_income": round(income / n_months, 2),
            "avg_monthly_expense": round(expense / n_months, 2),
            "n_transactions": len(txns),
        }

    # ── Conversation history ────────────────────────────────
    def add_message(self, user_id: str, role: str, content: str, lang: str = "en"):
        self.db.execute("INSERT INTO messages (user_id,role,content,lang) VALUES (?,?,?,?)",
                         (user_id, role, content, lang))
        self.db.commit()

    def get_history(self, user_id: str, last_n: int = 10) -> List[Dict]:
        rows = self.db.execute("SELECT role, content, lang FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                                (user_id, last_n)).fetchall()
        return list(reversed([dict(r) for r in rows]))

    # ── Key-value scratch space (e.g. remembered goals) ─────
    def set(self, user_id: str, key: str, value):
        self.db.execute("INSERT OR REPLACE INTO kv_store (user_id,key,value,updated_at) VALUES (?,?,?,datetime('now'))",
                         (user_id, key, json.dumps(value, ensure_ascii=False)))
        self.db.commit()

    def get(self, user_id: str, key: str, default=None):
        row = self.db.execute("SELECT value FROM kv_store WHERE user_id=? AND key=?", (user_id, key)).fetchone()
        return json.loads(row["value"]) if row else default

    # ── Full profile ─────────────────────────────────────────
    def get_full_profile(self, user_id: str) -> Dict:
        user = self.get_or_create_user(user_id)
        debts = self.get_debts(user_id)
        summary = self.get_monthly_summary(user_id)
        history = self.get_history(user_id, last_n=6)
        total_debt = sum(d["principal"] for d in debts)
        total_min_emi = sum(d["min_payment"] for d in debts)
        monthly_income = user.get("monthly_income") or summary["avg_monthly_income"]
        return {
            "user_id": user_id, "language": user.get("language", "en"),
            "monthly_income": round(monthly_income, 2), "debts": debts,
            "total_debt": round(total_debt, 2), "total_emi": round(total_min_emi, 2),
            "txn_summary": summary, "recent_history": history,
            "debt_to_income_pct": round(total_min_emi / max(monthly_income, 1) * 100, 1),
        }
