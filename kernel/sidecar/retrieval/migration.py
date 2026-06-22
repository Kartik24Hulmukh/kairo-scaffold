"""
migration.py -- SQLite schema migration manager for the Kairo sidecar.

Tracks schema version in the 'schema_version' table inside kairo.db and
applies numbered migrations in sequence.  New migrations are added as
functions registered in MIGRATIONS dict below.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Callable

# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

def _migration_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add page_index column to chunks table if it is missing (v1 -> v2).

    The column was present in the original CREATE TABLE statement
    (bench/run_bench.py lines 117-129) but older databases created before
    the canonical schema may be missing it.
    """
    cursor = conn.cursor()
    # Check if the chunks table actually exists.
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
    if not cursor.fetchone():
        return
    # PRAGMA table_info returns one row per column; check by name.
    cursor.execute("PRAGMA table_info(chunks)")
    columns = {row[1] for row in cursor.fetchall()}
    if "page_index" not in columns:
        cursor.execute("ALTER TABLE chunks ADD COLUMN page_index INTEGER")


# Map from (from_version, to_version) to migration callable.
MIGRATIONS: dict[tuple[int, int], Callable[[sqlite3.Connection], None]] = {
    (1, 2): _migration_v1_to_v2,
}


# ---------------------------------------------------------------------------
# MigrationManager
# ---------------------------------------------------------------------------

class MigrationManager:
    """Manages SQLite schema versioning for a Kairo database file.

    Usage::

        mm = MigrationManager()
        mm.run_migrations("/path/to/.kairo/kairo.db", target_version=2)
    """

    # Name of the table that stores the single schema version row.
    _VERSION_TABLE = "schema_version"

    def _ensure_version_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(  # nosemgrep
            f"""
            CREATE TABLE IF NOT EXISTS {self._VERSION_TABLE} (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        # Insert the initial row if the table is empty.
        conn.execute(  # nosemgrep
            f"INSERT OR IGNORE INTO {self._VERSION_TABLE} (id, version) VALUES (1, 1)"
        )
        conn.commit()

    def get_schema_version(self, db_path: str) -> int:
        """Return the current schema version stored in *db_path*.

        Creates the version-tracking table with version=1 if the database
        is new or pre-migration.

        Args:
            db_path: Absolute or relative path to the SQLite database file.

        Returns:
            Integer schema version (>= 1).
        """
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            self._ensure_version_table(conn)
            row = conn.execute(  # nosemgrep
                f"SELECT version FROM {self._VERSION_TABLE} WHERE id = 1"
            ).fetchone()
            return int(row[0]) if row else 1
        finally:
            conn.close()

    def run_migrations(self, db_path: str, target_version: int) -> None:
        """Apply all pending migrations from current version up to *target_version*.

        Migrations are applied one step at a time in ascending order.  Each
        step is wrapped in a transaction; failure rolls back that step only.

        Args:
            db_path: Absolute or relative path to the SQLite database file.
            target_version: The desired schema version after migrations complete.

        Raises:
            ValueError: If a required migration step is not registered in MIGRATIONS.
        """
        current = self.get_schema_version(db_path)
        if current >= target_version:
            return

        conn = sqlite3.connect(db_path)
        try:
            self._ensure_version_table(conn)
            for step in range(current, target_version):
                key = (step, step + 1)
                if key not in MIGRATIONS:
                    raise ValueError(
                        f"No migration registered for schema version {step} -> {step + 1}"
                    )
                migration_fn = MIGRATIONS[key]
                # Each migration step runs in its own transaction.
                conn.execute("BEGIN")
                try:
                    migration_fn(conn)
                    conn.execute(  # nosemgrep
                        f"UPDATE {self._VERSION_TABLE} SET version = ? WHERE id = 1",
                        (step + 1,),
                    )
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
        finally:
            conn.close()
