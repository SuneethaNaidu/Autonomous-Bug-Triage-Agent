import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "bug_triage.db"

def connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                issue_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                component TEXT,
                reporter TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'Open',
                severity TEXT,
                priority TEXT,
                assignee TEXT,
                category TEXT,
                duplicate_of TEXT,
                similarity REAL DEFAULT 0,
                explanation TEXT,
                classification_confidence REAL DEFAULT 0,
                review_status TEXT DEFAULT 'Pending',
                review_reason TEXT,
                result_json TEXT
            )
        """)

def save_issue(issue: dict[str, Any]):
    fields = [
        "issue_id", "title", "description", "component", "reporter",
        "created_at", "status", "severity", "priority", "assignee",
        "category", "duplicate_of", "similarity", "explanation",
        "classification_confidence", "review_status", "review_reason",
        "result_json"
    ]
    values = []
    for field in fields:
        value = issue.get(field)
        if field == "result_json" and not isinstance(value, str):
            value = json.dumps(value or {}, ensure_ascii=False)
        values.append(value)
    placeholders = ", ".join(["?"] * len(fields))
    columns = ", ".join(fields)
    with connect() as conn:
        conn.execute(
            f"INSERT INTO issues ({columns}) VALUES ({placeholders})",
            values
        )

def get_issues():
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM issues ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["result_json"] = json.loads(item.get("result_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            item["result_json"] = {}
        result.append(item)
    return result

def update_review(issue_id, review_status, assignee, severity, priority):
    with connect() as conn:
        conn.execute("""
            UPDATE issues
            SET review_status = ?, assignee = ?, severity = ?, priority = ?
            WHERE issue_id = ?
        """, (review_status, assignee, severity, priority, issue_id))

def issue_id_exists(issue_id):
    with connect() as conn:
        return conn.execute(
            "SELECT 1 FROM issues WHERE issue_id = ?", (issue_id,)
        ).fetchone() is not None

def clear_issues():
    with connect() as conn:
        conn.execute("DELETE FROM issues")
