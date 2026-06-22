//! macOS (WKWebView) overlay platform implementation.
//!
//! Quirks handled:
//!   - Transparency: remove NSWindow title bar + set backgroundColor to clear.
//!     Add NSVisualEffectView with HudWindow appearance for glass effect.
//!     Must be done on the main thread via `dispatch_sync`.
//!   - Click-through: `NSWindow.ignoresMouseEvents = YES` passes all events through.
//!     For pinned (interactive) mode: `ignoresMouseEvents = NO`.
//!   - WKWebView CORS: same as WebView2, but WKWebView may enforce stricter
//!     same-origin for custom protocols. Setting `Access-Control-Allow-Origin: *`
//!     on the response is required.
//!   - NSWindowLevel: use `.floating` (3) for always-on-top so it survives
//!     Exposé / Mission Control correctly. Tauri's set_always_on_top maps to this.
//!
//! Uses Tauri's raw-window-handle + objc2 for NSWindow manipulation.
//! The objc2 crate is only compiled on macOS via Cargo.toml conditional dep.

use tauri::{AppHandle, WebviewWindow};
use super::OverlayPlatform;

pub struct MacOsPlatform;

impl MacOsPlatform {
    pub fn new() -> Self {
        MacOsPlatform
    }
}

impl OverlayPlatform for MacOsPlatform {
    fn apply_transparency(&self, window: &WebviewWindow) {
        // Step 1: Use Tauri's set_transparent for the WebView layer.
        if let Err(e) = window.set_transparent(true) {
            eprintln!("[overlay/macos] set_transparent failed: {e}");
        }

        // Step 2: Remove decorations so the NSWindow has no title bar chrome.
        if let Err(e) = window.set_decorations(false) {
            eprintln!("[overlay/macos] set_decorations(false) failed: {e}");
        }

        // Step 3: Apply NSVisualEffectView (glass material) via raw NS handle.
        // We use objc2 for safe Objective-C interop.
        #[cfg(target_os = "macos")]
        apply_visual_effect_view(window);
    }

    fn set_click_through(&self, window: &WebviewWindow, enabled: bool) {
        // Tauri v2 set_ignore_cursor_events wraps NSWindow.ignoresMouseEvents on macOS.
        if let Err(e) = window.set_ignore_cursor_events(enabled) {
            eprintln!("[overlay/macos] set_ignore_cursor_events({enabled}) failed: {e}");
        }
    }

    fn set_pinned(&self, window: &WebviewWindow, pinned: bool) {
        if let Err(e) = window.set_always_on_top(pinned) {
            eprintln!("[overlay/macos] set_always_on_top({pinned}) failed: {e}");
        }

        // macOS: additionally set NSWindowLevel.floating so the window
        // appears above full-screen spaces correctly.
        #[cfg(target_os = "macos")]
        set_window_level_floating(window, pinned);
    }

    fn asset_protocol_headers(&self) -> Vec<(&'static str, &'static str)> {
        // WKWebView requires explicit Content-Type and CORS headers.
        // Some macOS versions enforce same-origin for custom schemes without these.
        vec![
            ("Content-Type", "image/png"),
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET"),
            ("Cache-Control", "public, max-age=3600"),
        ]
    }

    fn on_startup(&self, _app: &AppHandle) {
        // macOS: ensure the app doesn't appear in the Dock (overlay UX pattern).
        // This requires NSApp.setActivationPolicy(.accessory) which hides the Dock icon.
        #[cfg(target_os = "macos")]
        {
            use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
            use objc2::rc::autoreleasepool;
            autoreleasepool(|_| {
                let ns_app = unsafe { NSApplication::sharedApplication() };
                unsafe {
                    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Accessory);
                }
            });
        }
    }
}

/// Apply NSVisualEffectView behind the WebView for a glass/blur effect.
/// Must be called from main thread (Tauri's setup() closure runs on main thread).
#[cfg(target_os = "macos")]
fn apply_visual_effect_view(window: &WebviewWindow) {
    use objc2::rc::autoreleasepool;
    use objc2_app_kit::{
        NSView, NSVisualEffectView, NSVisualEffectMaterial,
        NSVisualEffectBlendingMode, NSVisualEffectState,
    };
    use raw_window_handle::HasWindowHandle;

    let Ok(handle) = window.window_handle() else { return };
    use raw_window_handle::RawWindowHandle;
    let RawWindowHandle::AppKit(app_kit_handle) = handle.as_raw() else { return };

    autoreleasepool(|_| {
        unsafe {
            // ns_view is the root WKWebView NSView
            let ns_view_ptr = app_kit_handle.ns_view.as_ptr();
            let ns_view: &NSView = &*(ns_view_ptr as *const NSView);
            let ns_window = ns_view.window().expect("view has no window");

            // Make window background clear
            ns_window.setOpaque(false);
            ns_window.setBackgroundColor(None);

            // Create NSVisualEffectView
            let frame = ns_view.frame();
            let vfx = NSVisualEffectView::initWithFrame(
                NSVisualEffectView::alloc(),
                frame,
            );
            vfx.setMaterial(NSVisualEffectMaterial::HudWindow);
            vfx.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
            vfx.setState(NSVisualEffectState::Active);
            vfx.setAutoresizingMask(
                objc2_app_kit::NSViewWidthSizable | objc2_app_kit::NSViewHeightSizable,
            );

            // Insert below the WebView content
            if let Some(superview) = ns_view.superview() {
                superview.addSubview_positioned_relativeTo(
                    &vfx,
                    objc2_app_kit::NSWindowBelow,
                    Some(ns_view),
                );
            }
        }
    });
}

/// Set NSWindowLevel to floating (3) for pinned mode or normal (0) otherwise.
#[cfg(target_os = "macos")]
fn set_window_level_floating(window: &WebviewWindow, pinned: bool) {
    use objc2_app_kit::NSWindowLevel;
    use raw_window_handle::HasWindowHandle;
    use raw_window_handle::RawWindowHandle;

    let Ok(handle) = window.window_handle() else { return };
    let RawWindowHandle::AppKit(app_kit_handle) = handle.as_raw() else { return };

    unsafe {
        use objc2::rc::autoreleasepool;
        use objc2_app_kit::NSView;
        autoreleasepool(|_| {
            let ns_view_ptr = app_kit_handle.ns_view.as_ptr();
            let ns_view: &NSView = &*(ns_view_ptr as *const NSView);
            if let Some(win) = ns_view.window() {
                let level = if pinned {
                    NSWindowLevel::Floating
                } else {
                    NSWindowLevel::Normal
                };
                win.setLevel(level);
            }
        });
    }
}
