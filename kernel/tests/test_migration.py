"""
test_migration.py -- unit tests for kernel/sidecar/retrieval/migration.py

Covers:
  - get_schema_version returns 1 for a brand-new database
  - run_migrations v1->v2 adds page_index column when missing
  - run_migrations is idempotent when version is already at target
  - run_migrations raises ValueError for an unregistered migration step
  - version is persisted across reconnections
"""

import os
import sqlite3
import tempfile
import pytest

from kernel.sidecar.retrieval.migration import MigrationManager


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_db_with_chunks_no_page_index(path: str) -> None:
    """Create a minimal chunks table that is missing the page_index column."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _column_names(db_path: str, table: str) -> set:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(f"PRAGMA table_info({table})")
    names = {row[1] for row in cursor.fetchall()}
    conn.close()
    return names


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestGetSchemaVersion:
    def test_new_database_returns_version_1(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        mm = MigrationManager()
        assert mm.get_schema_version(db) == 1

    def test_version_persists_across_reconnections(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        mm = MigrationManager()
        # First call initialises at version 1.
        mm.get_schema_version(db)
        # Second call on the same file must return the same value.
        assert mm.get_schema_version(db) == 1

    def test_creates_parent_dirs_if_missing(self, tmp_path):
        db = str(tmp_path / "nested" / "deep" / "kairo.db")
        mm = MigrationManager()
        # Should not raise even if intermediate dirs are absent.
        version = mm.get_schema_version(db)
        assert version == 1
        assert os.path.exists(db)


class TestRunMigrations:
    def test_v1_to_v2_adds_page_index_column(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        _make_db_with_chunks_no_page_index(db)
        mm = MigrationManager()
        # Initialise version table at 1.
        mm.get_schema_version(db)

        mm.run_migrations(db, target_version=2)

        assert "page_index" in _column_names(db, "chunks")

    def test_version_increments_after_migration(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        _make_db_with_chunks_no_page_index(db)
        mm = MigrationManager()
        mm.get_schema_version(db)
        mm.run_migrations(db, target_version=2)
        assert mm.get_schema_version(db) == 2

    def test_idempotent_when_already_at_target(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        _make_db_with_chunks_no_page_index(db)
        mm = MigrationManager()
        mm.get_schema_version(db)
        mm.run_migrations(db, target_version=2)
        # Running again must not raise and must leave version at 2.
        mm.run_migrations(db, target_version=2)
        assert mm.get_schema_version(db) == 2

    def test_no_op_when_target_equals_current(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        mm = MigrationManager()
        mm.get_schema_version(db)  # version = 1
        # Target == current should be a no-op.
        mm.run_migrations(db, target_version=1)
        assert mm.get_schema_version(db) == 1

    def test_raises_for_unregistered_migration(self, tmp_path):
        db = str(tmp_path / "kairo.db")
        mm = MigrationManager()
        mm.get_schema_version(db)  # version = 1
        # No migration registered for 1->99.
        with pytest.raises(ValueError, match="No migration registered"):
            mm.run_migrations(db, target_version=99)

    def test_page_index_column_not_duplicated_when_already_present(self, tmp_path):
        """Migration v1->v2 must not error if page_index already exists."""
        db = str(tmp_path / "kairo.db")
        # Create chunks table that already has page_index.
        conn = sqlite3.connect(db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                page_index INTEGER,
                text TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        mm = MigrationManager()
        mm.get_schema_version(db)
        # Should not raise even though column is already present.
        mm.run_migrations(db, target_version=2)
        assert "page_index" in _column_names(db, "chunks")
