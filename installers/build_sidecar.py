"""Sidecar PyInstaller builder script.

Runs PyInstaller to compile kernel/sidecar/app.py into a self-contained binary.
"""

import os
import subprocess
import sys
import pathlib

def main() -> None:
    repo_root = pathlib.Path(__file__).parent.parent
    spec_path = repo_root / "installers" / "kairo_sidecar.spec"
    
    # Locate pyinstaller in the virtualenv
    if os.name == "nt":
        pyinstaller_bin = repo_root / "kernel" / "sidecar" / ".venv" / "Scripts" / "pyinstaller.exe"
    else:
        pyinstaller_bin = repo_root / "kernel" / "sidecar" / ".venv" / "bin" / "pyinstaller"

    if not pyinstaller_bin.exists():
        print(f"Error: PyInstaller not found at {pyinstaller_bin}. Please run 'make setup-venv' first.", file=sys.stderr)
        sys.exit(1)

    print(f"==> Building sidecar using spec: {spec_path}")
    try:
        subprocess.check_call(
            [str(pyinstaller_bin), "--clean", "--noconfirm", str(spec_path)],
            cwd=str(repo_root)
        )
        print("==> PyInstaller build finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during PyInstaller build: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify output exists
    binary_ext = ".exe" if os.name == "nt" else ""
    output_bin = repo_root / "dist" / f"kairo-sidecar{binary_ext}"
    if output_bin.exists():
        print(f"==> SUCCESS: Sidecar binary created at: {output_bin}")
        sys.exit(0)
    else:
        print(f"==> ERROR: Output binary not found at: {output_bin}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
