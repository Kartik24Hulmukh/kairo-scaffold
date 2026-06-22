"""Tests for scripts/check_perf_budget.py."""

import http.server
import json
import os
import pathlib
import sys
import threading
import time
from typing import Generator
import pytest

# Make scripts/ and kernel/ importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.check_perf_budget import find_binary, check_perf_structure, BUDGETS

class MockSidecarHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logging

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "sidecar": "ok",
                "db_writable": True,
                "qdrant_available": False,
                "embedding_model": "hash_embed",
                "db_path": "mock.db"
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/index":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # return 10 pages indexed
            self.wfile.write(json.dumps({
                "doc_id": "mock_doc_123",
                "pages": 10,
                "chunks": 50
            }).encode())
        elif self.path == "/extract":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # return list of extractions
            self.wfile.write(json.dumps([]).encode())
        else:
            self.send_response(404)
            self.end_headers()

@pytest.fixture
def mock_server() -> Generator[str, None, None]:
    server = http.server.HTTPServer(("127.0.0.1", 7438), MockSidecarHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield "http://127.0.0.1:7438"
    server.shutdown()
    server.server_close()

def test_find_binary(tmp_path):
    # Setup mock root
    root = tmp_path
    assert find_binary(root) is None

    # Create dummy binary
    dist_dir = root / "dist"
    dist_dir.mkdir()
    dummy_bin = dist_dir / "kairo-sidecar.exe"
    dummy_bin.write_bytes(b"dummy exe content")

    found = find_binary(root)
    assert found is not None
    assert found.name == "kairo-sidecar.exe"

def test_check_perf_structure(tmp_path):
    root = tmp_path
    
    # Missing app.py
    checks = check_perf_structure(root)
    assert checks["sidecar_exists"] is False

    # Create dummy app.py
    sidecar_dir = root / "kernel" / "sidecar"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "app.py").touch()

    checks = check_perf_structure(root)
    assert checks["sidecar_exists"] is True
