"""Versioned migration runner for kairo.db (SQLite).

This is the Python-side mirror of ``overlay/src-tauri/src/migration.rs``.
Both must agree on SCHEMA_VERSION_CURRENT and migration semantics.

Design principles:
  - Schema version lives in ``PRAGMA user_version`` (built-in SQLite field).
  - Migrations are forward-only and idempotent (safe to re-run).
  - Each migration runs in a single transaction: ROLLBACK on failure.
  - A backup copy is created before any migration: ``kairo.db.bak.v<N>.<ts>``.
  - The backup is preserved even on success (user safety net).
  - Qdrant vector store versioning lives in a companion ``meta.json``:
    ``{"embedding_dim": N, "model": "...", "schema_version": N}``.
    If model/dim changes, a rebuild is triggered from SQLite chunk text.
  - A migration that fails leaves user_version unchanged and the original
    store fully usable.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

log = logging.getLogger("kairo.migration")

# ── Version constants ────────────────────────────────────────────────────────

#: Current application schema version. Increment whenever a new migration is added.
#: Must match the Rust constant in ``migration.rs``.
SCHEMA_VERSION_CURRENT: int = 2

#: Default embedding model name written to Qdrant meta.json.
#: Matches the fallback _HashEmbedder used when sentence_transformers is absent.
DEFAULT_EMBEDDING_MODEL: str = "hash-256"

#: Default embedding dimension for the _HashEmbedder.
DEFAULT_EMBEDDING_DIM: int = 256


# ── Migration definition ─────────────────────────────────────────────────────

@dataclass
class Migration:
    """A single forward migration step."""

    to_version: int
    """Target schema version after this migration completes."""

    description: str
    """Human-readable description for logging and audit trail."""

    statements: List[str]
    """SQL statements to execute. Each is idempotent (uses IF NOT EXISTS, etc.)."""


def _col_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return True if `column` exists in `table`."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _make_statements_v0_to_v1() -> List[str]:
    """Build migration statements for v0 → v1.

    Uses Python-side column-existence check so the SQL itself stays idempotent.
    """
    return [
        # Audit log table (always safe — IF NOT EXISTS)
        """CREATE TABLE IF NOT EXISTS schema_migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applied_at INTEGER NOT NULL,
            from_version INTEGER NOT NULL,
            to_version INTEGER NOT NULL,
            description TEXT NOT NULL
        )""",
        # documents.use_visual_grading — checked at runtime before issuing
        # (handled in _run_statement_idempotent via 'duplicate column name' catch)
        "ALTER TABLE documents ADD COLUMN use_visual_grading INTEGER NOT NULL DEFAULT 0",
    ]


def _make_statements_v1_to_v2() -> List[str]:
    return [
        """CREATE TABLE IF NOT EXISTS schema_migration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applied_at INTEGER NOT NULL,
            from_version INTEGER NOT NULL,
            to_version INTEGER NOT NULL,
            description TEXT NOT NULL
        )""",
        "ALTER TABLE pages ADD COLUMN dpi REAL NOT NULL DEFAULT 150.0",
        "ALTER TABLE extractions ADD COLUMN pack_name TEXT NOT NULL DEFAULT 'generic'",
    ]


#: All migrations in ascending version order.
MIGRATIONS: List[Migration] = [
    Migration(
        to_version=1,
        description="Establish schema_version tracking; add documents.use_visual_grading",
        statements=_make_statements_v0_to_v1(),
    ),
    Migration(
        to_version=2,
        description="Add pages.dpi; add extractions.pack_name",
        statements=_make_statements_v1_to_v2(),
    ),
]


# ── Public API ───────────────────────────────────────────────────────────────

class MigrationError(RuntimeError):
    """Raised when a migration fails. The database is left unchanged."""

    def __init__(self, message: str, to_version: Optional[int] = None):
        super().__init__(message)
        self.to_version = to_version


class VersionAheadError(MigrationError):
    """Raised when the database schema version is newer than the app."""

    def __init__(self, db_version: int, app_version: int):
        super().__init__(
            f"Database schema v{db_version} is newer than this app (v{app_version}). "
            f"Upgrade the app to open this database.",
        )
        self.db_version = db_version
        self.app_version = app_version


def run_migrations(db_path: pathlib.Path) -> None:
    """Run all pending migrations on the database at ``db_path``.

    Raises:
        MigrationError: if a migration SQL fails. Database is unchanged.
        VersionAheadError: if db schema is newer than app.
    """
    if not db_path.exists():
        log.debug("Database does not exist yet at %s — skipping migration.", db_path)
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        current_version = _get_user_version(conn)

        if current_version == SCHEMA_VERSION_CURRENT:
            log.debug("Schema is up to date (v%d). No migrations needed.", SCHEMA_VERSION_CURRENT)
            return

        if current_version > SCHEMA_VERSION_CURRENT:
            raise VersionAheadError(current_version, SCHEMA_VERSION_CURRENT)

        log.info(
            "Database is at v%d, app requires v%d. Running %d migration(s).",
            current_version,
            SCHEMA_VERSION_CURRENT,
            SCHEMA_VERSION_CURRENT - current_version,
        )

        # Create backup BEFORE touching anything.
        _create_backup(db_path, current_version)

        pending = [m for m in MIGRATIONS if m.to_version > current_version]
        for migration in pending:
            _run_single_migration(conn, migration, current_version)
            current_version = migration.to_version  # track for audit log

        log.info("All migrations complete. Schema is now v%d.", SCHEMA_VERSION_CURRENT)

    finally:
        conn.close()


def qdrant_needs_rebuild(
    qdrant_dir: pathlib.Path,
    expected_dim: int = DEFAULT_EMBEDDING_DIM,
    expected_model: str = DEFAULT_EMBEDDING_MODEL,
) -> bool:
    """Return True if the Qdrant vector store needs a rebuild.

    A rebuild is needed when:
      - ``meta.json`` exists but records a different ``embedding_dim`` or ``model``.
      - ``meta.json`` is corrupt/unreadable.

    A rebuild is NOT needed when:
      - ``meta.json`` does not exist (fresh store — will be created on first use).
      - ``meta.json`` records matching dim and model.
    """
    meta_path = qdrant_dir / "meta.json"
    if not meta_path.exists():
        return False  # Fresh store — no rebuild needed.

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim_ok = meta.get("embedding_dim") == expected_dim
        model_ok = meta.get("model") == expected_model
        return not (dim_ok and model_ok)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Could not read Qdrant meta.json: %s — assuming rebuild needed.", e)
        return True


def write_qdrant_meta(
    qdrant_dir: pathlib.Path,
    dim: int,
    model: str,
    schema_version: int = SCHEMA_VERSION_CURRENT,
) -> None:
    """Write a new Qdrant meta.json after a successful rebuild."""
    meta = {
        "embedding_dim": dim,
        "model": model,
        "schema_version": schema_version,
    }
    meta_path = qdrant_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Qdrant meta.json written: dim=%d, model=%s, schema_version=%d", dim, model, schema_version)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _get_user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version = {version}")


def _create_backup(db_path: pathlib.Path, old_version: int) -> pathlib.Path:
    ts = int(time.time())
    bak_name = f"{db_path.name}.bak.v{old_version}.{ts}"
    bak_path = db_path.parent / bak_name
    shutil.copy2(str(db_path), str(bak_path))
    log.info("Backup created at %s", bak_path)
    return bak_path


def _run_single_migration(
    conn: sqlite3.Connection,
    migration: Migration,
    from_version: int,
) -> None:
    """Execute one migration inside a transaction. Raises MigrationError on failure."""
    log.info(
        "Applying migration → v%d: %s",
        migration.to_version,
        migration.description,
    )

    try:
        conn.execute("BEGIN IMMEDIATE")

        for stmt in migration.statements:
            _run_statement_idempotent(conn, stmt)

        # Audit log (best-effort; table may not exist in first v0→v1 run before CREATE).
        ts = int(time.time())
        try:
            conn.execute(
                "INSERT INTO schema_migration_log "
                "(applied_at, from_version, to_version, description) VALUES (?, ?, ?, ?)",
                (ts, from_version, migration.to_version, migration.description),
            )
        except sqlite3.OperationalError:
            pass  # Audit log table didn't exist yet — created in this migration.

        # Set user_version within the same transaction.
        conn.execute(f"PRAGMA user_version = {migration.to_version}")

        conn.execute("COMMIT")
        log.info("✓ Migration to v%d succeeded.", migration.to_version)

    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise MigrationError(
            f"Migration to v{migration.to_version} failed (rolled back): {exc}",
            to_version=migration.to_version,
        ) from exc


def _run_statement_idempotent(conn: sqlite3.Connection, sql: str) -> None:
    """Execute a SQL statement, treating 'duplicate column name' as a no-op."""
    try:
        conn.execute(sql)
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        if "duplicate column name" in msg or "already exists" in msg:
            log.debug("Idempotent no-op (column already exists): %s", e)
        else:
            raise
