"""MIDAS Storage & Full-Text Search Layer.

Stores validated document extractions in SQLite with FTS5 virtual table
indexing for real-time querying across parameters, invoice vendors, and checklists.
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = "outputs/midas.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite tables and FTS5 full-text search indexes."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            source_file TEXT NOT NULL UNIQUE,
            extracted_json TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            status TEXT DEFAULT 'validated',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS rejections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            raw_extraction TEXT,
            rejection_reason TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
        USING fts5(doc_type, source_file, extracted_json, content='documents', content_rowid='id');

        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, doc_type, source_file, extracted_json)
            VALUES (new.id, new.doc_type, new.source_file, new.extracted_json);
        END;

        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_type, source_file, extracted_json)
            VALUES ('delete', old.id, old.doc_type, old.source_file, old.extracted_json);
        END;

        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, doc_type, source_file, extracted_json)
            VALUES ('delete', old.id, old.doc_type, old.source_file, old.extracted_json);
            INSERT INTO documents_fts(rowid, doc_type, source_file, extracted_json)
            VALUES (new.id, new.doc_type, new.source_file, new.extracted_json);
        END;
    """)

    con.commit()
    con.close()


def store_document(
    doc_type: str,
    source_file: str,
    data: Dict[str, Any],
    confidence: float,
    db_path: str = DB_PATH
) -> int:
    """Store or update a validated document record in SQLite.

    Idempotent on source_file.
    """
    init_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    json_str = json.dumps(data)
    cur.execute("""
        INSERT INTO documents (doc_type, source_file, extracted_json, confidence, status, created_at)
        VALUES (?, ?, ?, ?, 'validated', datetime('now'))
        ON CONFLICT(source_file) DO UPDATE SET
            doc_type = excluded.doc_type,
            extracted_json = excluded.extracted_json,
            confidence = excluded.confidence,
            status = 'validated',
            created_at = datetime('now')
    """, (doc_type, source_file, json_str, confidence))

    doc_id = cur.lastrowid
    con.commit()
    con.close()
    return doc_id


def store_rejection(
    doc_type: str,
    source_file: str,
    raw_extraction: Dict[str, Any],
    reason: str,
    db_path: str = DB_PATH
) -> int:
    """Store a rejected document record with error diagnostics."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        INSERT INTO rejections (doc_type, source_file, raw_extraction, rejection_reason, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
    """, (doc_type, source_file, json.dumps(raw_extraction), reason))

    rej_id = cur.lastrowid
    con.commit()
    con.close()
    return rej_id


def search(
    query: str,
    doc_type: Optional[str] = None,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """Execute full-text search query across all stored document extractions."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Clean query for FTS5 syntax
    clean_query = query.strip()
    if not clean_query:
        return list_documents(doc_type=doc_type, db_path=db_path)

    # Wrap multi-word query in quotes or append prefix match *
    safe_query = f'"{clean_query}"*' if " " in clean_query else f"{clean_query}*"

    try:
        if doc_type:
            sql = """
                SELECT d.* FROM documents d
                JOIN documents_fts f ON d.id = f.rowid
                WHERE documents_fts MATCH ? AND d.doc_type = ?
                ORDER BY d.id DESC
            """
            rows = con.execute(sql, (safe_query, doc_type)).fetchall()
        else:
            sql = """
                SELECT d.* FROM documents d
                JOIN documents_fts f ON d.id = f.rowid
                WHERE documents_fts MATCH ?
                ORDER BY d.id DESC
            """
            rows = con.execute(sql, (safe_query,)).fetchall()
    except sqlite3.OperationalError:
        # Fallback to LIKE query if FTS expression is unusual
        pattern = f"%{clean_query}%"
        if doc_type:
            rows = con.execute(
                "SELECT * FROM documents WHERE extracted_json LIKE ? AND doc_type = ? ORDER BY id DESC",
                (pattern, doc_type)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM documents WHERE extracted_json LIKE ? ORDER BY id DESC",
                (pattern,)
            ).fetchall()

    results = []
    for r in rows:
        item = dict(r)
        item["extracted_data"] = json.loads(item["extracted_json"])
        results.append(item)

    con.close()
    return results


def get_document(doc_id: int, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Fetch a single document record by primary key ID."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    con.close()
    if not row:
        return None
    res = dict(row)
    res["extracted_data"] = json.loads(res["extracted_json"])
    return res


def list_documents(
    doc_type: Optional[str] = None,
    limit: int = 50,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """List most recently processed documents."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    if doc_type:
        rows = con.execute(
            "SELECT * FROM documents WHERE doc_type = ? ORDER BY id DESC LIMIT ?",
            (doc_type, limit)
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM documents ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()

    con.close()
    results = []
    for r in rows:
        item = dict(r)
        item["extracted_data"] = json.loads(item["extracted_json"])
        results.append(item)
    return results


def get_metrics(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Retrieve pipeline operational metrics and extraction statistics."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    total_docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    total_rejections = cur.execute("SELECT COUNT(*) FROM rejections").fetchone()[0]
    avg_conf = cur.execute("SELECT AVG(confidence) FROM documents").fetchone()[0] or 0.0

    by_type = {}
    rows = cur.execute("SELECT doc_type, COUNT(*), AVG(confidence) FROM documents GROUP BY doc_type").fetchall()
    for dt, cnt, a_conf in rows:
        by_type[dt] = {
            "count": cnt,
            "avg_confidence": round(a_conf or 0.0, 3)
        }

    con.close()
    total_attempts = total_docs + total_rejections
    rejection_rate = (total_rejections / total_attempts) if total_attempts > 0 else 0.0

    return {
        "total_documents": total_docs,
        "total_rejections": total_rejections,
        "overall_avg_confidence": round(avg_conf, 3),
        "rejection_rate": round(rejection_rate, 3),
        "by_type": by_type,
    }
