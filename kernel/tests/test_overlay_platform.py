"""
test_overlay_platform.py -- unit tests for Kairo overlay platform configuration and security contracts
"""

import json
import os
import pathlib
import re
import sys
import pytest

# Make sure repo root is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent


def test_tauri_conf_properties():
    """Verify that tauri.conf.json has the correct overlay window flags set."""
    conf_path = REPO_ROOT / "overlay" / "src-tauri" / "tauri.conf.json"
    assert conf_path.exists(), f"tauri.conf.json not found at {conf_path}"

    with open(conf_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # In Tauri v2, windows are configured under app.windows array
    windows = config.get("app", {}).get("windows", [])
    assert len(windows) > 0, "No windows defined in tauri.conf.json"

    main_win = next((w for w in windows if w.get("label") == "main"), None)
    assert main_win is not None, "Main window configuration not found"

    # Crucial overlay flags
    assert main_win.get("decorations") is False, "Window must not have decorations (title bar)"
    assert main_win.get("transparent") is True, "Window must be transparent"
    assert main_win.get("alwaysOnTop") is True, "Window must be pinned (alwaysOnTop)"


def test_cargo_toml_dependencies():
    """Verify that Cargo.toml has platform-specific target dependencies."""
    cargo_path = REPO_ROOT / "overlay" / "src-tauri" / "Cargo.toml"
    assert cargo_path.exists(), f"Cargo.toml not found at {cargo_path}"

    with open(cargo_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify target blocks exist for cross-platform dependencies
    assert 'cfg(target_os = "windows")' in content, "Cargo.toml is missing target block for windows"
    assert 'cfg(target_os = "macos")' in content, "Cargo.toml is missing target block for macos"
    assert 'windows = ' in content, "Cargo.toml is missing windows dependency"
    assert 'objc2 = ' in content, "Cargo.toml is missing objc2 dependency"
    assert 'objc2-app-kit = ' in content, "Cargo.toml is missing objc2-app-kit dependency"


# ---------------------------------------------------------------------------
# kairo-img:// Security Contract Simulation
# ---------------------------------------------------------------------------

def simulate_serve_asset(uri_path: str) -> int:
    """
    Python simulation of serve_asset's security validation from asset_protocol.rs.
    Returns the HTTP status code (200, 400, 404).
    """
    # Rust: let filename = path.strip_prefix('/').unwrap_or(path);
    # let sha256 = filename.strip_suffix(".png").unwrap_or(filename);
    filename = uri_path.lstrip('/')
    if filename.endswith(".png"):
        sha256 = filename[:-4]
    else:
        sha256 = filename

    # Security check:
    # Rust: sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) || sha256.contains(['/', '\\', '.', '\0'])
    if (len(sha256) != 64 or 
        not all(c in "0123456789abcdefABCDEF" for c in sha256) or 
        any(c in sha256 for c in ('/', '\\', '.', '\0'))):
        return 400  # BAD_REQUEST

    # Simulating file existence check
    mock_images_dir = REPO_ROOT / ".kairo" / "page_images"
    img_path = mock_images_dir / f"{sha256}.png"
    
    if not img_path.exists():
        return 404  # NOT_FOUND

    return 200  # OK


def test_kairo_img_security_contract(tmp_path):
    """Test the SHA256 validation logic of the custom asset protocol to prevent directory traversal."""
    # 1. Valid hex SHA256
    valid_sha = "a" * 64
    assert simulate_serve_asset(f"/{valid_sha}.png") in (200, 404)

    # 2. Too short / too long SHA256
    assert simulate_serve_asset("/" + "a" * 63 + ".png") == 400
    assert simulate_serve_asset("/" + "a" * 65 + ".png") == 400

    # 3. Non-hex characters
    assert simulate_serve_asset("/" + "g" * 64 + ".png") == 400
    assert simulate_serve_asset("/" + "a" * 63 + "z.png") == 400

    # 4. Path traversal attempts
    assert simulate_serve_asset(f"/../{valid_sha}.png") == 400
    assert simulate_serve_asset(f"/\\\\{valid_sha}.png") == 400
    assert simulate_serve_asset(f"/{valid_sha}/subfile.png") == 400
    assert simulate_serve_asset(f"/{valid_sha}\0.png") == 400


def test_asset_protocol_headers():
    """Verify that headers returned for asset protocol contain required CORS header."""
    # Check windows.rs and macos.rs headers configuration
    windows_path = REPO_ROOT / "overlay" / "src-tauri" / "src" / "platform" / "windows.rs"
    macos_path = REPO_ROOT / "overlay" / "src-tauri" / "src" / "platform" / "macos.rs"

    assert windows_path.exists()
    assert macos_path.exists()

    with open(windows_path, "r", encoding="utf-8") as f:
        windows_content = f.read()

    with open(macos_path, "r", encoding="utf-8") as f:
        macos_content = f.read()

    # Must contain Access-Control-Allow-Origin
    assert "Access-Control-Allow-Origin" in windows_content
    assert "Access-Control-Allow-Origin" in macos_content
