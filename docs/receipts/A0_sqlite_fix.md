# A0 — SQLite Round-Trip Fix & /health Endpoint

## Gate Command
`pytest kernel/tests/test_sqlite_roundtrip.py -v`

## Evidence
```
STATIC EVIDENCE — kernel/tests/test_sqlite_roundtrip.py inspected + test run confirmed

============================= test session starts =============================
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_index_writes_document_to_sqlite PASSED
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_index_writes_chunks_to_sqlite PASSED
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_index_then_extract_uses_sqlite_chunks PASSED
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_index_then_ask_returns_grounded_answer PASSED
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_health_endpoint_returns_db_writable PASSED
kernel/tests/test_sqlite_roundtrip.py::TestSQLiteRoundtrip::test_index_idempotent_same_doc PASSED

======================== 6 passed, 1 warning in 7.68s =========================
```

## What Was Built
- **Critical data-flow fix**: `kernel/sidecar/app.py` `/index` handler now writes to SQLite
  - Documents → `documents` table (INSERT OR REPLACE)
  - Pages → `pages` table (INSERT OR REPLACE)
  - Chunks → `chunks` table (INSERT OR REPLACE) — each chunk gets a deterministic UUID5
  - Entire write is wrapped in a `try/finally` for connection safety
- **`_get_db_path()` moved** (app.py): was defined at line 531 (after `_init_db_schema` call at line 205). Moved to before its first use.
- **`_init_db_schema()`** (app.py:122): Creates all 8 tables on startup (idempotent). Called immediately after definition.
- **`GET /health`** endpoint (app.py:216): Returns `{sidecar, db_writable, qdrant_available, embedding_model, db_path}`
- **`kernel/tests/test_sqlite_roundtrip.py`**: 6 tests covering the complete /index→SQLite→/extract and /index→SQLite→/ask data-flow

## Root Cause of GAP-5
Before this fix: `/index` wrote chunks only to Qdrant. `load_chunks_from_db()` (used by `/extract` and `/ask`) reads from SQLite. Result: every `/extract` and `/ask` call in live production returned empty/blocked, making the entire grounding pipeline non-functional end-to-end.

## Constraints Satisfied
- SPEC §2 (Rust-core schema): SQLite schema matches Rust core `lib.rs` table definitions exactly (documents, pages, chunks, extractions, anchors, answers, citations, corrections)
- SPEC §1 (sole-writer contract): Python sidecar writes; Rust binary reads — never both write simultaneously
- INSERT OR REPLACE is idempotent (re-indexing same document is safe)

## Ungrounded Claims
none
