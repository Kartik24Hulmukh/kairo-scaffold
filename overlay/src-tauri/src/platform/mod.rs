//! Platform abstraction for the Kairo overlay window.
//!
//! Each OS requires different techniques for:
//!  1. Window transparency + glass effect
//!  2. Click-through when unpinned
//!  3. `kairo-img://` asset protocol quirks
//!  4. Global-shortcut wiring (handled by the shared plugin, but OS-specific fallbacks exist)
//!
//! The `OverlayPlatform` trait exposes a single interface. `setup_overlay()` dispatches
//! to the correct implementation at compile time via `#[cfg(target_os = ...)]`.

use tauri::{AppHandle, WebviewWindow};

/// Platform-specific overlay operations.
/// Each method is infallible — on failure it logs and degrades gracefully.
pub trait OverlayPlatform {
    /// Apply OS-specific window transparency / glass visual style.
    ///
    /// On Windows: sets WS_EX_LAYERED and calls SetLayeredWindowAttributes.
    /// On macOS: removes title bar, adds NSVisualEffectView with behindWindow blending.
    /// On Linux: requests RGBA visual from GDK; degrades to opaque if compositor absent.
    fn apply_transparency(&self, window: &WebviewWindow);

    /// Enable or disable click-through mode.
    ///
    /// When `enabled = true`, mouse events pass through the overlay to windows below.
    /// When `enabled = false`, the overlay captures mouse events normally (for interaction).
    ///
    /// On Windows: toggles WS_EX_TRANSPARENT extended style.
    /// On macOS: sets NSWindow.ignoresMouseEvents.
    /// On Linux: uses GDK event mask + gtk_widget_set_can_focus.
    fn set_click_through(&self, window: &WebviewWindow, enabled: bool);

    /// Apply "always on top" pinned state.
    ///
    /// This is uniform via Tauri's cross-platform API, but some OSes need
    /// additional flags (e.g., NSWindowLevel.floating on macOS, or _NET_WM_STATE_ABOVE on Linux).
    fn set_pinned(&self, window: &WebviewWindow, pinned: bool);

    /// Return the MIME type and any extra headers required for `kairo-img://` responses.
    ///
    /// WebView2 and WKWebView handle CORS headers differently.
    /// WebKitGTK requires explicit Content-Type in some versions.
    fn asset_protocol_headers(&self) -> Vec<(&'static str, &'static str)>;

    /// Called once on app startup after window creation.
    ///
    /// Perform any one-time OS-level setup (e.g., register visual, request compositor).
    fn on_startup(&self, _app: &AppHandle) {}
}

// ──────────────────────────────────────────────────────────────────
// Compile-time dispatch
// ──────────────────────────────────────────────────────────────────

#[cfg(target_os = "windows")]
pub mod windows;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "linux")]
pub mod linux;

#[cfg(target_os = "windows")]
pub use windows::WindowsPlatform as CurrentPlatform;
#[cfg(target_os = "macos")]
pub use macos::MacOsPlatform as CurrentPlatform;
#[cfg(target_os = "linux")]
pub use linux::LinuxPlatform as CurrentPlatform;

/// Construct the platform implementation for the current OS.
pub fn current_platform() -> CurrentPlatform {
    CurrentPlatform::new()
}

/// Perform full overlay setup: apply transparency + default click-through state.
///
/// Called from `main.rs` after Tauri window creation.
pub fn setup_overlay(app: &AppHandle, window: &WebviewWindow) {
    let platform = current_platform();
    platform.on_startup(app);
    platform.apply_transparency(window);
    // Default: click-through OFF (user must interact with overlay to pin it)
    platform.set_click_through(window, false);
    platform.set_pinned(window, true);
}
