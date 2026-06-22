import sys
import os
import time
import subprocess
import requests
from fastapi.testclient import TestClient

# Adjust path to import from repositories/kairo-scaffold
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.mock_models.mock_sidecar import app

def test_mock_cascade_determinism():
    # Set seed 42
    os.environ["KAIRO_MOCK_SEED"] = "42"
    client = TestClient(app)
    
    # 1. Run /index
    res1_seed42 = client.post("/index", json={"path": "dummy_path"})
    assert res1_seed42.status_code == 200
    data1_seed42 = res1_seed42.json()
    
    # 2. Run /index again and verify it is identical
    res2_seed42 = client.post("/index", json={"path": "dummy_path"})
    assert res2_seed42.status_code == 200
    data2_seed42 = res2_seed42.json()
    
    assert data1_seed42 == data2_seed42
    
    # 3. Change seed to 99 and verify the output is different
    os.environ["KAIRO_MOCK_SEED"] = "99"
    res_seed99 = client.post("/index", json={"path": "dummy_path"})
    assert res_seed99.status_code == 200
    data_seed99 = res_seed99.json()
    
    assert data1_seed42 != data_seed99
    
    # 4. Verify /extract is deterministic
    os.environ["KAIRO_MOCK_SEED"] = "42"
    ext1_seed42 = client.post("/extract", json={"doc_id": "doc_123", "pack": "invoice"}).json()
    ext2_seed42 = client.post("/extract", json={"doc_id": "doc_123", "pack": "invoice"}).json()
    assert ext1_seed42 == ext2_seed42
    
    os.environ["KAIRO_MOCK_SEED"] = "99"
    ext_seed99 = client.post("/extract", json={"doc_id": "doc_123", "pack": "invoice"}).json()
    assert ext1_seed42 != ext_seed99

    # 5. Verify /ask is deterministic
    os.environ["KAIRO_MOCK_SEED"] = "42"
    ask1_seed42 = client.post("/ask", json={"doc_id": "doc_123", "query": "What is Kairo?"}).json()
    ask2_seed42 = client.post("/ask", json={"doc_id": "doc_123", "query": "What is Kairo?"}).json()
    assert ask1_seed42 == ask2_seed42
    
    os.environ["KAIRO_MOCK_SEED"] = "99"
    ask_seed99 = client.post("/ask", json={"doc_id": "doc_123", "query": "What is Kairo?"}).json()
    assert ask1_seed42 != ask_seed99
    
    # 6. Verify refusal is deterministic
    refuse_ans = client.post("/ask", json={"doc_id": "doc_123", "query": "unanswerable question"}).json()
    assert refuse_ans["grounded"] is False
    assert refuse_ans["text"] == "blocked"

def test_mock_sidecar_server_port():
    # Start the server on port 7439
    env = os.environ.copy()
    env["KAIRO_MOCK_SEED"] = "123"
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "tests.mock_models.mock_sidecar:app", 
        "--port", "7439", 
        "--host", "127.0.0.1"
    ], env=env)
    
    # Wait for server to boot up (max 5 seconds)
    url = "http://127.0.0.1:7439/docs"
    success = False
    for _ in range(50):
        try:
            r = requests.get(url, timeout=0.5)
            if r.status_code == 200:
                success = True
                break
        except Exception:
            pass
        time.sleep(0.1)
        
    try:
        assert success, "Mock sidecar did not boot on port 7439"
        
        # Test index endpoint
        r = requests.post("http://127.0.0.1:7439/index", json={"path": "dummy_path"})
        assert r.status_code == 200
        data = r.json()
        assert "doc_id" in data
        assert "seed 123" in data["chunks_list"][0]["text"]
    finally:
        proc.terminate()
        proc.wait()
