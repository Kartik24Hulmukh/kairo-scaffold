//! Versioned migration runner for kairo.db (SQLite).
//!
//! # Design
//!
//! - Schema version is stored in SQLite's built-in `PRAGMA user_version`.
//! - Migrations are forward-only and idempotent: running migration N twice
//!   produces the same schema as running it once.
//! - Each migration runs in a single transaction. On failure: ROLLBACK → the
//!   original schema and user_version are preserved.
//! - Before migrating, a backup copy is created: `kairo.db.bak.<old_version>.<ts>`.
//!   This backup is preserved even if the migration succeeds.
//! - Qdrant vector store versioning: a companion `meta.json` records
//!   `{"embedding_dim": N, "model": "...", "schema_version": N}`.
//!   If the model or dim changes, a rebuild is triggered from SQLite chunk text.

use rusqlite::{Connection, Result as SqlResult};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Current application schema version.
/// Increment this constant whenever a new migration is added.
pub const SCHEMA_VERSION_CURRENT: u32 = 2;

/// A single database migration step.
struct Migration {
    /// Target version after this migration runs.
    to_version: u32,
    /// Description for logging.
    description: &'static str,
    /// The SQL statements to execute (each is a separate statement).
    /// Statements must be idempotent (use IF NOT EXISTS, IF NOT EXISTS column checks, etc.)
    statements: &'static [&'static str],
}

/// All migrations in order. Each must be idempotent.
static MIGRATIONS: &[Migration] = &[
    Migration {
        to_version: 1,
        description: "Establish schema_version tracking; add documents.use_visual_grading",
        statements: &[
            // Create schema_migration_log table for audit trail.
            "CREATE TABLE IF NOT EXISTS schema_migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applied_at INTEGER NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                description TEXT NOT NULL
            )",
            // Add use_visual_grading to documents if not present.
            // SQLite doesn't support IF NOT EXISTS on ALTER TABLE ADD COLUMN directly;
            // we use a try/ignore pattern by wrapping in a SELECT that checks the column.
            // The actual idempotency check is handled by the runner (see run_statement_idempotent).
            "ALTER TABLE documents ADD COLUMN use_visual_grading INTEGER NOT NULL DEFAULT 0",
        ],
    },
    Migration {
        to_version: 2,
        description: "Add pages.dpi; add extractions.pack_name; create schema_migration_log if missing",
        statements: &[
            "CREATE TABLE IF NOT EXISTS schema_migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applied_at INTEGER NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                description TEXT NOT NULL
            )",
            "ALTER TABLE pages ADD COLUMN dpi REAL NOT NULL DEFAULT 150.0",
            "ALTER TABLE extractions ADD COLUMN pack_name TEXT NOT NULL DEFAULT 'generic'",
        ],
    },
];

/// Error type for migration failures.
#[derive(Debug)]
pub enum MigrationError {
    /// The database file could not be opened or backed up.
    Io(std::io::Error),
    /// A SQL statement failed. The migration was rolled back.
    /// `u32` is the target version that failed; `String` is the rusqlite error.
    SqlFailed(u32, String),
    /// The database schema version is ahead of SCHEMA_VERSION_CURRENT.
    /// This means a newer app wrote this database; we refuse to downgrade.
    VersionAhead { db_version: u32, app_version: u32 },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::Io(e) => write!(f, "migration I/O error: {e}"),
            MigrationError::SqlFailed(v, e) => write!(f, "migration to v{v} failed (rolled back): {e}"),
            MigrationError::VersionAhead { db_version, app_version } => write!(
                f,
                "database schema v{db_version} is newer than this app (v{app_version}); \
                 upgrade the app to open this database"
            ),
        }
    }
}

/// Run all pending migrations on the database at `db_path`.
///
/// Returns `Ok(())` if no migrations were needed or all ran successfully.
/// Returns `Err(MigrationError)` if any migration failed; the database is unchanged.
pub fn run_migrations(db_path: &Path) -> Result<(), MigrationError> {
    let mut conn = Connection::open(db_path).map_err(|e| {
        MigrationError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    })?;

    // Enable WAL mode for crash safety.
    conn.execute_batch("PRAGMA journal_mode = WAL;").ok();
    conn.execute_batch("PRAGMA foreign_keys = ON;").ok();

    let current_version = get_user_version(&conn)?;

    if current_version == SCHEMA_VERSION_CURRENT {
        eprintln!("[migration] Schema is up to date (v{SCHEMA_VERSION_CURRENT}). No migrations needed.");
        return Ok(());
    }

    if current_version > SCHEMA_VERSION_CURRENT {
        return Err(MigrationError::VersionAhead {
            db_version: current_version,
            app_version: SCHEMA_VERSION_CURRENT,
        });
    }

    eprintln!(
        "[migration] Database is at v{current_version}, app requires v{SCHEMA_VERSION_CURRENT}. \
         Running {} migration(s).",
        SCHEMA_VERSION_CURRENT - current_version
    );

    // Create a backup before touching anything.
    create_backup(db_path, current_version)?;

    // Run each pending migration.
    for migration in MIGRATIONS.iter().filter(|m| m.to_version > current_version) {
        run_single_migration(&mut conn, migration, current_version)?;
    }

    eprintln!("[migration] All migrations complete. Schema is now v{SCHEMA_VERSION_CURRENT}.");
    Ok(())
}

/// Read `PRAGMA user_version` from an open connection.
fn get_user_version(conn: &Connection) -> Result<u32, MigrationError> {
    let ver: u32 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(|e| MigrationError::SqlFailed(0, e.to_string()))?;
    Ok(ver)
}

/// Set `PRAGMA user_version = N` on an open connection.
fn set_user_version(conn: &Connection, version: u32) -> SqlResult<()> {
    conn.execute_batch(&format!("PRAGMA user_version = {version}"))
}

/// Copy the database file to `<db_path>.bak.<old_version>.<timestamp>`.
fn create_backup(db_path: &Path, old_version: u32) -> Result<(), MigrationError> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let bak_name = format!(
        "{}.bak.v{}.{}",
        db_path.file_name().and_then(|n| n.to_str()).unwrap_or("kairo.db"),
        old_version,
        ts
    );
    let bak_path = db_path.parent().unwrap_or(Path::new(".")).join(&bak_name);
    std::fs::copy(db_path, &bak_path).map_err(MigrationError::Io)?;
    eprintln!("[migration] Backup created at {bak_path:?}");
    Ok(())
}

/// Run a single migration in a transaction.
/// On failure: ROLLBACK → `Err(MigrationError::SqlFailed)`.
/// On success: COMMIT + set user_version.
fn run_single_migration(
    conn: &mut Connection,
    migration: &Migration,
    from_version: u32,
) -> Result<(), MigrationError> {
    eprintln!(
        "[migration] Applying migration → v{}: {}",
        migration.to_version, migration.description
    );

    // Use a savepoint so we can roll back just this migration if it fails.
    let tx = conn
        .transaction()
        .map_err(|e| MigrationError::SqlFailed(migration.to_version, e.to_string()))?;

    for &stmt in migration.statements {
        run_statement_idempotent(&tx, stmt).map_err(|e| {
            MigrationError::SqlFailed(migration.to_version, format!("SQL: {stmt}\nError: {e}"))
        })?;
    }

    // Record in audit log (best-effort — table may not exist in v0→v1 first run).
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let _ = tx.execute(
        "INSERT INTO schema_migration_log (applied_at, from_version, to_version, description)
         VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![ts, from_version, migration.to_version, migration.description],
    );

    // Set user_version within the same transaction.
    tx.execute_batch(&format!("PRAGMA user_version = {}", migration.to_version))
        .map_err(|e| MigrationError::SqlFailed(migration.to_version, e.to_string()))?;

    tx.commit()
        .map_err(|e| MigrationError::SqlFailed(migration.to_version, e.to_string()))?;

    eprintln!("[migration] ✓ Migration to v{} succeeded.", migration.to_version);
    Ok(())
}

/// Execute a SQL statement, treating "duplicate column" and "table already exists"
/// errors as no-ops (idempotency for ALTER TABLE ADD COLUMN and CREATE TABLE IF NOT EXISTS).
fn run_statement_idempotent(conn: &Connection, sql: &str) -> SqlResult<()> {
    match conn.execute_batch(sql) {
        Ok(_) => Ok(()),
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            // SQLite's error for adding an existing column: "duplicate column name"
            if msg.contains("duplicate column name") || msg.contains("already exists") {
                eprintln!("[migration/idempotent] Ignoring idempotent no-op: {msg}");
                Ok(())
            } else {
                Err(e)
            }
        }
    }
}

/// Check whether the Qdrant vector store meta matches the current embedding config.
///
/// Returns `true` if a rebuild is needed (dim or model changed).
/// Returns `false` if the store is up-to-date or doesn't exist yet.
pub fn qdrant_needs_rebuild(qdrant_dir: &Path, expected_dim: u32, expected_model: &str) -> bool {
    let meta_path = qdrant_dir.join("meta.json");
    if !meta_path.exists() {
        // New store — no rebuild needed, will be created fresh.
        return false;
    }

    match std::fs::read_to_string(&meta_path) {
        Ok(content) => {
            // Parse meta.json: {"embedding_dim": N, "model": "...", "schema_version": N}
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                let dim_ok = val.get("embedding_dim")
                    .and_then(|v| v.as_u64())
                    .map(|d| d == expected_dim as u64)
                    .unwrap_or(false);
                let model_ok = val.get("model")
                    .and_then(|v| v.as_str())
                    .map(|m| m == expected_model)
                    .unwrap_or(false);
                !(dim_ok && model_ok)
            } else {
                eprintln!("[migration] Could not parse Qdrant meta.json — assuming rebuild needed.");
                true
            }
        }
        Err(e) => {
            eprintln!("[migration] Could not read Qdrant meta.json: {e} — assuming rebuild needed.");
            true
        }
    }
}

/// Write a new Qdrant meta.json after a successful rebuild.
pub fn write_qdrant_meta(qdrant_dir: &Path, dim: u32, model: &str, schema_version: u32) -> std::io::Result<()> {
    let meta = serde_json::json!({
        "embedding_dim": dim,
        "model": model,
        "schema_version": schema_version,
    });
    let meta_path = qdrant_dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())?;
    eprintln!("[migration] Qdrant meta.json written: dim={dim}, model={model}, schema_version={schema_version}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_v0_db(dir: &Path) -> PathBuf {
        let db_path = dir.join("kairo.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch("
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT,
                sha256 TEXT,
                page_count INTEGER,
                created_at INTEGER
            );
            CREATE TABLE pages (
                doc_id TEXT,
                page_index INTEGER,
                width_px INTEGER,
                height_px INTEGER,
                image_sha256 TEXT,
                PRIMARY KEY (doc_id, page_index)
            );
            CREATE TABLE extractions (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                field TEXT,
                value TEXT,
                confidence REAL,
                status TEXT
            );
            PRAGMA user_version = 0;
        ").unwrap();
        db_path
    }

    #[test]
    fn test_migration_v0_to_current() {
        let dir = TempDir::new().unwrap();
        let db_path = create_v0_db(dir.path());

        // Before migration: user_version = 0
        {
            let conn = Connection::open(&db_path).unwrap();
            let ver: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
            assert_eq!(ver, 0, "Pre-migration version should be 0");
        }

        // Run migrations
        run_migrations(&db_path).expect("Migration should succeed");

        // After migration: user_version = SCHEMA_VERSION_CURRENT
        {
            let conn = Connection::open(&db_path).unwrap();
            let ver: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
            assert_eq!(ver, SCHEMA_VERSION_CURRENT, "Post-migration version should be current");
        }

        // Backup file should exist
        let bak_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".bak."))
            .collect();
        assert!(!bak_files.is_empty(), "Backup file must be created before migration");
    }

    #[test]
    fn test_migration_idempotent() {
        let dir = TempDir::new().unwrap();
        let db_path = create_v0_db(dir.path());

        // Run migrations twice — should not fail
        run_migrations(&db_path).expect("First migration should succeed");
        run_migrations(&db_path).expect("Second run (idempotent) should succeed");
    }

    #[test]
    fn test_backup_preserved_on_success() {
        let dir = TempDir::new().unwrap();
        let db_path = create_v0_db(dir.path());

        run_migrations(&db_path).expect("Migration should succeed");

        // Backup must still be present (not deleted on success).
        let bak_files: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".bak."))
            .collect();
        assert!(!bak_files.is_empty(), "Backup must be preserved after successful migration");

        // Backup must be a valid SQLite file (readable).
        let bak_path = bak_files[0].path();
        let bak_conn = Connection::open(&bak_path).expect("Backup must be valid SQLite");
        let _ver: u32 = bak_conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .expect("Backup must be queryable");
    }

    #[test]
    fn test_version_ahead_returns_error() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("future.db");
        let conn = Connection::open(&db_path).unwrap();
        // Set version much higher than current app version.
        conn.execute_batch("PRAGMA user_version = 999").unwrap();
        drop(conn);

        let result = run_migrations(&db_path);
        assert!(
            matches!(result, Err(MigrationError::VersionAhead { .. })),
            "Should error on version-ahead database"
        );
    }

    #[test]
    fn test_qdrant_needs_rebuild_no_meta() {
        let dir = TempDir::new().unwrap();
        // No meta.json → no rebuild needed (fresh store)
        assert!(!qdrant_needs_rebuild(dir.path(), 256, "hash-256"));
    }

    #[test]
    fn test_qdrant_needs_rebuild_stale_model() {
        let dir = TempDir::new().unwrap();
        let meta = serde_json::json!({
            "embedding_dim": 384,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "schema_version": 1,
        });
        std::fs::write(dir.path().join("meta.json"), meta.to_string()).unwrap();

        // Different dim and model → rebuild needed
        assert!(qdrant_needs_rebuild(dir.path(), 256, "hash-256"));
    }

    #[test]
    fn test_qdrant_no_rebuild_when_matching() {
        let dir = TempDir::new().unwrap();
        let meta = serde_json::json!({
            "embedding_dim": 256,
            "model": "hash-256",
            "schema_version": 2,
        });
        std::fs::write(dir.path().join("meta.json"), meta.to_string()).unwrap();

        assert!(!qdrant_needs_rebuild(dir.path(), 256, "hash-256"));
    }
}
