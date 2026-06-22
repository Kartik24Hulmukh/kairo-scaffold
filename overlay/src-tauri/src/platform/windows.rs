//! Windows (WebView2) overlay platform implementation.
//!
//! Quirks handled:
//!   - Transparency requires `transparent: true` in tauri.conf.json AND
//!     the WS_EX_LAYERED extended window style.  Tauri sets the conf flag;
//!     we set the style here for belt-and-suspenders compatibility.
//!   - Click-through: toggle WS_EX_TRANSPARENT on the HWND.
//!     WS_EX_TRANSPARENT alone makes the window transparent to mouse BUT
//!     still painted — combine with WS_EX_LAYERED for the full effect.
//!   - WebView2 CORS: `Access-Control-Allow-Origin: *` is sufficient.
//!   - DWM Blur-Behind: not used (too many driver-level incompatibilities);
//!     we rely on the WebView2 transparent background instead.
//!
//! winapi is only linked on Windows via [target.'cfg(target_os="windows")'.dependencies]
//! in Cargo.toml.

use tauri::{AppHandle, WebviewWindow};
use super::OverlayPlatform;

pub struct WindowsPlatform;

impl WindowsPlatform {
    pub fn new() -> Self {
        WindowsPlatform
    }
}

impl OverlayPlatform for WindowsPlatform {
    fn apply_transparency(&self, window: &WebviewWindow) {
        // Apply WS_EX_LAYERED via raw HWND for belt-and-suspenders on older WebView2.
        // This is safe to call even if Tauri already set it.
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::UI::WindowsAndMessaging::{
                GetWindowLongPtrW, SetWindowLongPtrW,
                GWL_EXSTYLE, WS_EX_LAYERED,
            };
            use windows::Win32::Foundation::HWND;

            if let Ok(hwnd_ptr) = window.hwnd() {
                let hwnd = HWND(hwnd_ptr.0);
                unsafe {
                    let ex_style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
                    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex_style | WS_EX_LAYERED.0 as isize);
                }
            }
        }
    }

    fn set_click_through(&self, window: &WebviewWindow, enabled: bool) {
        // Tauri v2 provides set_ignore_cursor_events() cross-platform.
        if let Err(e) = window.set_ignore_cursor_events(enabled) {
            eprintln!("[overlay/windows] set_ignore_cursor_events({enabled}) failed: {e}");
        }
    }

    fn set_pinned(&self, window: &WebviewWindow, pinned: bool) {
        if let Err(e) = window.set_always_on_top(pinned) {
            eprintln!("[overlay/windows] set_always_on_top({pinned}) failed: {e}");
        }
    }

    fn asset_protocol_headers(&self) -> Vec<(&'static str, &'static str)> {
        // WebView2 requires Access-Control-Allow-Origin for custom schemes.
        vec![
            ("Content-Type", "image/png"),
            ("Access-Control-Allow-Origin", "*"),
            ("Cache-Control", "public, max-age=3600"),
        ]
    }

    fn on_startup(&self, _app: &AppHandle) {
        // Enable DPI awareness — WebView2 handles this internally, but
        // the host process should also be DPI-aware to avoid blur on HiDPI.
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::UI::HiDpi::SetProcessDpiAwarenessContext;
            use windows::Win32::UI::HiDpi::DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2;
            unsafe {
                let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
            }
        }
    }
}
