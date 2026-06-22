//! Unified `kairo-img://` asset protocol handler.
//!
//! This module extracts the protocol handler from `main.rs` so it can be
//! registered from a single place, with OS-specific headers applied via
//! the current `OverlayPlatform` implementation.
//!
//! URL format: `kairo-img://<sha256>.png`
//!   - `<sha256>` must be exactly 64 lowercase hex characters
//!   - File served from `.kairo/page_images/<sha256>.png`
//!
//! Security: the sha256 is validated before any filesystem access.
//! Path traversal is prevented by rejecting any non-hex character in the name.

use tauri::http::{Response, StatusCode};
use std::path::PathBuf;

use crate::platform::OverlayPlatform;
use crate::platform::current_platform;

/// Serve a single `kairo-img://` request.
///
/// Called from `Builder::register_uri_scheme_protocol` on all OSes.
/// `kairo_dir` is the resolved path to the `.kairo/` directory.
pub fn serve_asset(request: &tauri::http::Request<Vec<u8>>, kairo_dir: &PathBuf) -> Response<Vec<u8>> {
    let platform = current_platform();
    let headers = platform.asset_protocol_headers();

    let path = request.uri().path();
    let filename = path.strip_prefix('/').unwrap_or(path);
    let sha256 = filename.strip_suffix(".png").unwrap_or(filename);

    // Validate: exactly 64 hex chars, no path separators, no null bytes.
    if sha256.len() != 64
        || !sha256.chars().all(|c| c.is_ascii_hexdigit())
        || sha256.contains(['/', '\\', '.', '\0'])
    {
        return build_response(StatusCode::BAD_REQUEST, Vec::new(), &headers);
    }

    let img_path = kairo_dir.join("page_images").join(format!("{sha256}.png"));

    if !img_path.exists() {
        return build_response(StatusCode::NOT_FOUND, Vec::new(), &headers);
    }

    match std::fs::read(&img_path) {
        Ok(bytes) => build_response(StatusCode::OK, bytes, &headers),
        Err(e) => {
            eprintln!("[asset_protocol] Failed to read {img_path:?}: {e}");
            build_response(StatusCode::INTERNAL_SERVER_ERROR, Vec::new(), &headers)
        }
    }
}

/// Build an HTTP response with the given status, body, and platform-specific headers.
fn build_response(
    status: StatusCode,
    body: Vec<u8>,
    headers: &[(&'static str, &'static str)],
) -> Response<Vec<u8>> {
    let mut builder = Response::builder().status(status);
    for (key, value) in headers {
        builder = builder.header(*key, *value);
    }
    builder.body(body).unwrap()
}

/// Closure suitable for passing to `Builder::register_uri_scheme_protocol`.
///
/// Example usage in `main.rs`:
/// ```rust
/// use crate::asset_protocol;
/// ...
/// .register_uri_scheme_protocol("kairo-img", |_app, request| {
///     let kairo_dir = find_kairo_dir().unwrap_or_default();
///     asset_protocol::serve_asset(request, &kairo_dir)
/// })
/// ```
pub fn make_handler(
) -> impl Fn(&tauri::AppHandle, &tauri::http::Request<Vec<u8>>) -> Response<Vec<u8>> + Send + Sync + 'static
{
    move |_app, request| {
        // Re-resolve kairo dir on every request in case CWD changed (hot-reload scenarios).
        let kairo_dir = crate::find_kairo_dir().unwrap_or_else(|_| PathBuf::from(".kairo"));
        serve_asset(request, &kairo_dir)
    }
}
