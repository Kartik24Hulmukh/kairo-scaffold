from fastapi.testclient import TestClient
import sys
import os

# Adjust path to import from kernel.sidecar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.app import app

client = TestClient(app)

def test_index(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, this is a test document.\nIt has multiple paragraphs.")
    
    import hashlib
    content = test_file.read_bytes()
    expected_doc_id = hashlib.sha256(content).hexdigest()
    
    res = client.post("/index", json={"path": str(test_file)})
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == expected_doc_id[:16]
    assert data["pages"] == 1
    assert data["chunks"] > 0
    assert isinstance(data["pages"], int)
    assert isinstance(data["chunks"], int)
    assert "pages_list" in data
    assert "chunks_list" in data
    assert len(data["pages_list"]) == 1
    assert len(data["chunks_list"]) > 0

def test_index_pdf():
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/test.pdf"))
    
    import hashlib
    with open(pdf_path, "rb") as f:
        content = f.read()
    expected_doc_id = hashlib.sha256(content).hexdigest()
    
    res = client.post("/index", json={"path": pdf_path})
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == expected_doc_id[:16]
    assert data["pages"] == 1
    assert data["chunks"] == 2
    assert isinstance(data["pages"], int)
    assert isinstance(data["chunks"], int)
    assert len(data["pages_list"]) == 1
    assert len(data["chunks_list"]) == 2
    assert data["chunks_list"][0]["text"] == "Left PDF"
    assert data["chunks_list"][1]["text"] == "Right PDF"

def test_index_docx():
    docx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/test_track.docx"))
    
    import hashlib
    with open(docx_path, "rb") as f:
        content = f.read()
    expected_doc_id = hashlib.sha256(content).hexdigest()
    
    res = client.post("/index", json={"path": docx_path})
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == expected_doc_id[:16]
    assert data["pages"] == 1
    assert data["chunks"] == 0
    assert len(data["pages_list"]) == 1
    assert len(data["chunks_list"]) == 0

def test_index_txt():
    txt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/sample_contract_01.txt"))
    
    import hashlib
    with open(txt_path, "rb") as f:
        content = f.read()
    expected_doc_id = hashlib.sha256(content).hexdigest()
    
    res = client.post("/index", json={"path": txt_path})
    assert res.status_code == 200
    data = res.json()
    assert data["doc_id"] == expected_doc_id[:16]
    assert data["pages"] == 1
    assert data["chunks"] > 0
    assert len(data["pages_list"]) == 1
    assert len(data["chunks_list"]) > 0

def test_extract():
    res = client.post("/extract", json={"doc_id": "doc_123", "pack": "invoice"})
    assert res.status_code == 200
    assert res.json() == []

def test_ask():
    res = client.post("/ask", json={"doc_id": "doc_123", "query": "What is Kairo?"})
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "ans_123"
    assert data["text"] == "Stub answer"
    assert data["grounded"] is True

def test_provenance():
    res = client.get("/provenance/ext_123")
    assert res.status_code == 200
    data = res.json()
    assert data["page"] == 1
    assert data["image_ref"] == "img_123"

def test_correct():
    res = client.post("/correct", json={"extraction_id": "ext_123", "new_value": "new_val"})
    assert res.status_code == 200
    data = res.json()
    assert data["extraction_id"] == "ext_123"
    assert data["new_value"] == "new_val"
    assert data["by"] == "user"

def test_health():
    """GET /health returns sidecar status (supports kairo doctor command)."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    # Health response: {sidecar, db_writable, qdrant_available, embedding_model, db_path}
    assert data["sidecar"] in ("ok", "degraded")
    assert isinstance(data["db_writable"], bool)
    assert "embedding_model" in data

def test_ask_visual_unindexed_doc():
    """POST /ask/visual with unindexed doc_id returns empty result, not an error."""
    res = client.post("/ask/visual", json={
        "doc_id": "nonexistent-doc-xyz",
        "query": "What is in the table?",
        "page_index": 0,
        "top_k": 5,
    })
    assert res.status_code == 200
    data = res.json()
    # visual_retrieval_enabled=False for unindexed doc; matched_bbox=None
    assert "visual_retrieval_enabled" in data
    assert data["visual_retrieval_enabled"] is False
    assert data["matched_bbox"] is None
    assert data["iou_passed"] is False

def test_ask_visual_missing_query():
    """POST /ask/visual with missing required field returns 422."""
    res = client.post("/ask/visual", json={
        "doc_id": "some-doc",
        # missing query field
        "page_index": 0,
    })
    assert res.status_code == 422
