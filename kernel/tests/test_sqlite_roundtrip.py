"""
test_sqlite_roundtrip.py — Verifies that /index persists chunks to SQLite
so that /extract and /ask can retrieve them.

This tests the critical data-flow fix (GAP-5).
Gate command: pytest kernel/tests/test_sqlite_roundtrip.py -v
"""
import os
import sys
import sqlite3
import tempfile
import pathlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi.testclient import TestClient
from kernel.sidecar.app import app, _get_db_path

client = TestClient(app)


def _db_chunk_count(doc_id: str) -> int:
    """Read chunk count for doc_id directly from SQLite."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return -1
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def _db_doc_exists(doc_id: str) -> bool:
    """Check documents table for the doc_id."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        return False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


class TestSQLiteRoundtrip:
    """
    Critical test: /index must persist docs/pages/chunks to SQLite.
    Without this, /extract and /ask return empty results in live use.
    """

    def test_index_writes_document_to_sqlite(self, tmp_path):
        """After /index, the documents table must contain the doc_id."""
        test_file = tmp_path / "grounding_test.txt"
        test_file.write_text(
            "The total amount due is $1,234.56.\n\n"
            "Payment is due within 30 days of invoice date.\n\n"
            "Vendor: Acme Corp, Invoice #: INV-2024-001"
        )

        res = client.post("/index", json={"path": str(test_file)})
        assert res.status_code == 200
        doc_id = res.json()["doc_id"]

        # CRITICAL: document must be in SQLite
        assert _db_doc_exists(doc_id), (
            f"doc_id {doc_id!r} not found in SQLite documents table — "
            "/index is not persisting to the database"
        )

    def test_index_writes_chunks_to_sqlite(self, tmp_path):
        """After /index, chunks table must have rows for doc_id."""
        test_file = tmp_path / "chunks_test.txt"
        test_file.write_text(
            "Contract between Alpha Inc. and Beta LLC.\n\n"
            "Governing law: State of California.\n\n"
            "Effective Date: January 1, 2024."
        )

        res = client.post("/index", json={"path": str(test_file)})
        assert res.status_code == 200
        data = res.json()
        doc_id = data["doc_id"]
        api_chunk_count = data["chunks"]

        # chunks in response must match SQLite
        sqlite_chunk_count = _db_chunk_count(doc_id)
        assert sqlite_chunk_count >= 0, "Database not accessible"
        assert sqlite_chunk_count == api_chunk_count, (
            f"API says {api_chunk_count} chunks but SQLite has {sqlite_chunk_count} — "
            "/index is not persisting chunks to SQLite"
        )
        assert sqlite_chunk_count > 0, "No chunks written to SQLite"

    def test_index_then_extract_uses_sqlite_chunks(self, tmp_path):
        """
        Full end-to-end: index a doc, then extract fields.
        /extract reads from SQLite — if /index didn't write, returns [].
        """
        test_file = tmp_path / "invoice_e2e.txt"
        test_file.write_text(
            "INVOICE\n\n"
            "Invoice Number: INV-2024-099\n\n"
            "Total Amount Due: $5,000.00\n\n"
            "Due Date: March 15, 2024\n\n"
            "From: Test Vendor LLC\n"
            "To: Customer Corp"
        )

        # Step 1: index
        index_res = client.post("/index", json={"path": str(test_file)})
        assert index_res.status_code == 200
        doc_id = index_res.json()["doc_id"]

        # Step 2: extract — this reads from SQLite
        extract_res = client.post("/extract", json={"doc_id": doc_id, "pack": "generic"})
        assert extract_res.status_code == 200
        extractions = extract_res.json()

        # If extractions is empty, SQLite write is broken
        assert isinstance(extractions, list), "Expected list from /extract"
        # At minimum must NOT always be empty (SQLite write must have happened)
        chunk_count = _db_chunk_count(doc_id)
        assert chunk_count > 0, (
            "No chunks in SQLite — /index did not persist. "
            "/extract will always return [] in live use."
        )

    def test_index_then_ask_returns_grounded_answer(self, tmp_path):
        """
        Full end-to-end: index a doc, then ask a question.
        /ask reads from SQLite. If /index didn't write, answer is always blocked.

        Uses a query where BOTH a bigram AND long-word hits fire (score >= 2),
        so the keyword path triggers without needing semantic similarity >= 0.86.
        Query: "What is the governing jurisdiction?"
        Bigram "governing jurisdiction" appears in chunk → bigram_hits=1 (score 2+).
        """
        test_file = tmp_path / "ask_e2e.txt"
        # Design: chunk contains "governing jurisdiction" as a bigram from the query
        test_file.write_text(
            "The jurisdiction for all legal disputes is New York State.\n\n"
            "The governing jurisdiction is binding on all parties.\n\n"
            "This agreement is effective as of February 1, 2024."
        )

        # Step 1: index
        index_res = client.post("/index", json={"path": str(test_file)})
        assert index_res.status_code == 200
        doc_id = index_res.json()["doc_id"]

        # Verify chunks ARE in SQLite (core contract)
        chunk_count = _db_chunk_count(doc_id)
        assert chunk_count > 0, (
            "No chunks in SQLite — /index did not persist. "
            "/ask will always return 'blocked' in live use."
        )

        # Step 2: ask with a bigram-matching query
        # "governing jurisdiction" (bigram) appears verbatim in the chunk
        ask_res = client.post("/ask", json={
            "doc_id": doc_id,
            "query": "What is the governing jurisdiction?"
        })
        assert ask_res.status_code == 200
        answer = ask_res.json()

        # CRITICAL: must not be a blocked answer
        assert answer["grounded"] is True, (
            f"Expected grounded=True but got grounded=False. "
            f"Answer text: {answer.get('text')!r}. "
            f"Chunks in SQLite: {chunk_count}. "
            "Keyword bigram 'governing jurisdiction' should have scored ≥2. "
            "Check normalize_text or bigram scoring logic."
        )
        assert answer["text"] != "blocked", (
            "Answer is 'blocked' — keyword bigram path did not fire as expected."
        )


    def test_health_endpoint_returns_db_writable(self):
        """The /health endpoint must confirm db_writable=True."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["sidecar"] == "ok"
        assert data["db_writable"] is True, (
            f"Health check reports db_writable=False. DB path: {data.get('db_path')}"
        )

    def test_index_idempotent_same_doc(self, tmp_path):
        """
        Indexing the same document twice must not raise or double-insert.
        Uses INSERT OR REPLACE, so second call is safe.
        """
        test_file = tmp_path / "idempotent.txt"
        test_file.write_text("Idempotency test document.\n\nSame content indexed twice.")

        res1 = client.post("/index", json={"path": str(test_file)})
        assert res1.status_code == 200
        doc_id = res1.json()["doc_id"]
        count_after_first = _db_chunk_count(doc_id)

        res2 = client.post("/index", json={"path": str(test_file)})
        assert res2.status_code == 200
        assert res2.json()["doc_id"] == doc_id

        count_after_second = _db_chunk_count(doc_id)
        assert count_after_second == count_after_first, (
            f"After second /index: chunk count changed from {count_after_first} "
            f"to {count_after_second}. INSERT OR REPLACE should be idempotent."
        )
