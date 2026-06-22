// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod asset_protocol;
mod migration;
mod platform;
mod sidecar;

use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut, ShortcutState, GlobalShortcutExt};
use serde::Serialize;
use std::path::PathBuf;
use crate::platform::OverlayPlatform;

// ── DTOs ─────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct DocumentDto {
    doc_id: String,
    source_path: String,
}

#[derive(Serialize)]
struct PageMetadataDto {
    width_px: u32,
    height_px: u32,
    image_sha256: String,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Locate the `.kairo/` data directory by walking up from CWD.
pub fn find_kairo_dir() -> Result<PathBuf, String> {
    let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        let makefile = dir.join("Makefile");
        if makefile.exists() {
            let kairo_dir = dir.join(".kairo");
            if kairo_dir.exists() {
                return Ok(kairo_dir);
            }
        }
        if !dir.pop() {
            break;
        }
    }
    Err("Could not find Makefile or .kairo directory by traversing up".to_string())
}

fn find_kairo_db() -> Result<PathBuf, String> {
    let kairo_dir = find_kairo_dir()?;
    let db_path = kairo_dir.join("kairo.db");
    if db_path.exists() {
        Ok(db_path)
    } else {
        Err(format!("Database not found at {:?}", db_path))
    }
}

// ── Tauri commands ────────────────────────────────────────────────────────────

#[tauri::command]
fn get_documents() -> Result<Vec<DocumentDto>, String> {
    let db_path = find_kairo_db()?;
    let conn = rusqlite::Connection::open(db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT doc_id, source_path FROM documents")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |row| {
            Ok(DocumentDto {
                doc_id: row.get(0)?,
                source_path: row.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?;

    let mut docs = Vec::new();
    for row in rows {
        docs.push(row.map_err(|e| e.to_string())?);
    }
    Ok(docs)
}

#[tauri::command]
fn get_page_metadata(doc_id: String, page_index: usize) -> Result<PageMetadataDto, String> {
    let db_path = find_kairo_db()?;
    let conn = rusqlite::Connection::open(db_path).map_err(|e| e.to_string())?;
    let mut stmt = conn
        .prepare("SELECT width_px, height_px, image_sha256 FROM pages WHERE doc_id = ? AND page_index = ?")
        .map_err(|e| e.to_string())?;

    let page_meta = stmt.query_row([&doc_id, &page_index.to_string()], |row| {
        Ok(PageMetadataDto {
            width_px: row.get(0)?,
            height_px: row.get(1)?,
            image_sha256: row.get(2)?,
        })
    });

    match page_meta {
        Ok(meta) => Ok(meta),
        Err(e) => Err(e.to_string()),
    }
}

/// Tauri command: toggle click-through on the overlay.
/// Called from JS when user pins/unpins the overlay.
#[tauri::command]
fn set_overlay_click_through(app: tauri::AppHandle, enabled: bool) {
    let p = platform::current_platform();
    if let Some(window) = app.get_webview_window("main") {
        p.set_click_through(&window, enabled);
    }
}

/// Tauri command: toggle pinned (always-on-top) state.
#[tauri::command]
fn set_overlay_pinned(app: tauri::AppHandle, pinned: bool) {
    let p = platform::current_platform();
    if let Some(window) = app.get_webview_window("main") {
        p.set_pinned(&window, pinned);
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    // Run schema migrations before starting the Tauri app.
    // This ensures the Rust-side DB access (get_documents, get_page_metadata)
    // always works against the current schema.
    if let Ok(db_path) = find_kairo_db() {
        match migration::run_migrations(&db_path) {
            Ok(()) => {}
            Err(migration::MigrationError::VersionAhead { db_version, app_version }) => {
                eprintln!(
                    "[kairo] FATAL: Database schema v{db_version} is newer than this app (v{app_version}). \
                     Please upgrade Kairo."
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("[kairo] WARNING: Migration failed: {e}. Continuing with existing schema.");
            }
        }
    }

    // Build global shortcut plugin.
    let shortcut_plugin = tauri_plugin_global_shortcut::Builder::new()
        .with_handler(|app, _shortcut, event| {
            if event.state() == ShortcutState::Pressed {
                if let Some(window) = app.get_webview_window("main") {
                    if let Ok(visible) = window.is_visible() {
                        if visible {
                            let _ = window.hide();
                        } else {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                }
            }
        })
        .build();

    let app = tauri::Builder::default()
        .plugin(shortcut_plugin)
        .plugin(tauri_plugin_shell::init())
        // Register kairo-img:// unified asset protocol handler.
        .register_uri_scheme_protocol("kairo-img", |_context, request| {
            let kairo_dir = find_kairo_dir().unwrap_or_else(|_| PathBuf::from(".kairo"));
            asset_protocol::serve_asset(&request, &kairo_dir)
        })
        .setup(|app| {
            sidecar::start_sidecar(app)?;

            // Register global shortcut: Ctrl+Alt+Space.
            let ctrl_alt_space = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::ALT), Code::Space);
            app.global_shortcut().register(ctrl_alt_space)?;

            // Apply platform-specific overlay setup (transparency, click-through, pinning).
            if let Some(window) = app.get_webview_window("main") {
                platform::setup_overlay(app.handle(), &window);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_documents,
            get_page_metadata,
            set_overlay_click_through,
            set_overlay_pinned,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            sidecar::stop_sidecar(app_handle);
        }
    });
}
