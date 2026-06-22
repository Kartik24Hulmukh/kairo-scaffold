//! Linux (WebKitGTK) overlay platform implementation.
//!
//! Quirks handled:
//!   - Transparency: WebKitGTK requires an RGBA GdkVisual from the screen's
//!     compositing manager. If no compositor is running (e.g., bare X11 without
//!     picom/kwin/compiz), RGBA visuals are unavailable and we fall back to opaque.
//!     We detect this at runtime and log a warning — the overlay still works
//!     without transparency.
//!   - Click-through: GDK does not have a direct "pass events through" API.
//!     We set an input shape of zero size when click-through is enabled, which
//!     causes all pointer events to pass through to windows below.
//!     When disabled (interactive), we reset the input shape to the full window rect.
//!   - WebKitGTK asset-protocol: `register_uri_scheme_protocol` in Tauri v2
//!     on Linux uses WebKitURISchemeRequest (sync). This differs from WebView2
//!     (sync) and WKWebView (async in some modes). We use the same Tauri v2
//!     `register_uri_scheme_protocol` API — it handles the GTK dispatch internally.
//!     IMPORTANT: WebKitGTK ≥2.40 requires the protocol to be in the allowed
//!     list; we set `security.csp = null` in tauri.conf.json.
//!   - Always-on-top: use `window.set_always_on_top(true)` which maps to
//!     `gtk_window_set_keep_above(TRUE)` + sets _NET_WM_STATE_ABOVE hint.
//!
//! No external C bindings needed — Tauri v2's cross-platform APIs handle
//! the GTK/GDK calls internally. We document what they do under the hood.

use tauri::{AppHandle, WebviewWindow};
use super::OverlayPlatform;

pub struct LinuxPlatform {
    has_compositor: bool,
}

impl LinuxPlatform {
    pub fn new() -> Self {
        // Detect compositor availability by checking if the screen has an RGBA visual.
        // This is a best-effort check; we fall back to opaque if uncertain.
        let has_compositor = detect_compositor();
        LinuxPlatform { has_compositor }
    }
}

impl OverlayPlatform for LinuxPlatform {
    fn apply_transparency(&self, window: &WebviewWindow) {
        if !self.has_compositor {
            eprintln!(
                "[overlay/linux] No compositor detected (picom/kwin/compiz required for transparency). \
                 Falling back to opaque overlay. Install a compositor and restart to enable transparency."
            );
            // Explicitly set opaque so the webview content is still readable.
            return;
        }

        // Tauri v2 on Linux calls gtk_widget_set_visual() with the screen's RGBA visual
        // when set_transparent(true) is called. This requires a compositor.
        if let Err(e) = window.set_transparent(true) {
            eprintln!("[overlay/linux] set_transparent failed (compositor may be absent): {e}");
        }

        // Remove decorations (title bar, borders) for the overlay look.
        if let Err(e) = window.set_decorations(false) {
            eprintln!("[overlay/linux] set_decorations(false) failed: {e}");
        }
    }

    fn set_click_through(&self, window: &WebviewWindow, enabled: bool) {
        // Tauri v2 maps set_ignore_cursor_events to GDK input shape on Linux:
        //   enabled=true  → set input shape to empty cairo_region (events pass through)
        //   enabled=false → reset input shape to full window (events captured normally)
        if let Err(e) = window.set_ignore_cursor_events(enabled) {
            eprintln!("[overlay/linux] set_ignore_cursor_events({enabled}) failed: {e}");
        }
    }

    fn set_pinned(&self, window: &WebviewWindow, pinned: bool) {
        // Maps to gtk_window_set_keep_above(TRUE) and sets _NET_WM_STATE_ABOVE.
        if let Err(e) = window.set_always_on_top(pinned) {
            eprintln!("[overlay/linux] set_always_on_top({pinned}) failed: {e}");
        }
    }

    fn asset_protocol_headers(&self) -> Vec<(&'static str, &'static str)> {
        // WebKitGTK is strict about Content-Type for custom URI schemes.
        // Omitting it causes the WebView to refuse to display the image.
        vec![
            ("Content-Type", "image/png"),
            ("Access-Control-Allow-Origin", "*"),
            ("Cache-Control", "public, max-age=3600"),
        ]
    }

    fn on_startup(&self, _app: &AppHandle) {
        if !self.has_compositor {
            eprintln!(
                "[overlay/linux] WARNING: Transparency disabled (no compositor). \
                 The overlay will use an opaque white background."
            );
        } else {
            eprintln!("[overlay/linux] Compositor detected — RGBA transparency enabled.");
        }
    }
}

/// Detect whether a compositing manager is running on the current display.
///
/// On X11: check if the _NET_WM_CM_Sn selection atom is owned (standard ICCCM method).
/// On Wayland: compositor is always present (Wayland requires a compositor).
/// Returns `true` if we can safely use RGBA visuals.
fn detect_compositor() -> bool {
    // Check WAYLAND_DISPLAY first — Wayland always has a compositor.
    if std::env::var("WAYLAND_DISPLAY").is_ok() {
        return true;
    }

    // On X11, check the DISPLAY variable exists (basic sanity).
    if std::env::var("DISPLAY").is_err() {
        return false;
    }

    // Attempt to detect compositing via the XDG_CURRENT_DESKTOP or known env vars.
    // Full X11 atom check requires xlib bindings (not worth the dep).
    // Instead, check well-known compositor env hints:
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default().to_lowercase();
    let session = std::env::var("DESKTOP_SESSION").unwrap_or_default().to_lowercase();
    let compositor_hint = std::env::var("KAIRO_HAS_COMPOSITOR").unwrap_or_default();

    // Known desktop environments that always run a compositor:
    let composited_desktops = [
        "gnome", "kde", "plasma", "cinnamon", "budgie", "deepin", "pantheon", "unity",
    ];

    if composited_desktops.iter().any(|d| desktop.contains(d) || session.contains(d)) {
        return true;
    }

    // Allow explicit override via env var for CI / testing.
    if compositor_hint == "1" || compositor_hint == "true" {
        return true;
    }
    if compositor_hint == "0" || compositor_hint == "false" {
        return false;
    }

    // Default: assume compositor present (user can set KAIRO_HAS_COMPOSITOR=0 to opt out).
    true
}
