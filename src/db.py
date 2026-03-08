"""
Database helpers — SQLite storage for judged cases.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "data" / "cases.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                name TEXT,
                year INTEGER,
                case_type TEXT,
                facts TEXT,
                actual_verdict TEXT,
                ai_verdict TEXT,
                ai_confidence REAL,
                ai_reasoning TEXT,
                fairness_score REAL,
                fairness_notes TEXT,
                match BOOLEAN,
                judged_at TIMESTAMP
            )
        """)
        conn.commit()


def upsert_case(case: dict):
    """Insert or replace a case record."""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO cases (
                id, name, year, case_type, facts, actual_verdict,
                ai_verdict, ai_confidence, ai_reasoning,
                fairness_score, fairness_notes, match, judged_at
            ) VALUES (
                :id, :name, :year, :case_type, :facts, :actual_verdict,
                :ai_verdict, :ai_confidence, :ai_reasoning,
                :fairness_score, :fairness_notes, :match, :judged_at
            )
        """, case)
        conn.commit()


def get_all_cases() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM cases ORDER BY year").fetchall()
    return [dict(r) for r in rows]


def get_judged_cases() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE ai_verdict IS NOT NULL ORDER BY year"
        ).fetchall()
    return [dict(r) for r in rows]
