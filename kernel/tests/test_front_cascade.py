from fastapi.testclient import TestClient
import sys
import os
import pytest
import re

# Adjust path to import from kernel.sidecar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.app import app, load_chunks_from_db
from kernel.sidecar.ingest.quote_align import smith_waterman_align

client = TestClient(app)

def test_front_cascade_verbatim_quote_and_span(tmp_path):
    # 1. Create a dummy text document with distinct sentences
    test_file = tmp_path / "front_test.txt"
    test_file.write_text(
        "This is the first sentence of the dummy document. "
        "The total amount due is $450.00. "
        "Please pay by June 30, 2026."
    )
    
    # Ingest document
    res = client.post("/index", json={"path": str(test_file)})
    assert res.status_code == 200
    doc_id = res.json()["doc_id"]
    
    # 2. Ask a question that should match one specific sentence
    query = "What is the total amount due?"
    res_ask = client.post("/ask", json={"doc_id": doc_id, "query": query})
    assert res_ask.status_code == 200
    data = res_ask.json()
    
    # Answer should be grounded
    assert data["grounded"] is True
    # Under FRONT pattern, the answer text should only be the selected quote sentence, not the whole document/chunk!
    assert "The total amount due is $450.00." in data["text"]
    assert "Please pay by June 30, 2026." not in data["text"]
    assert "This is the first sentence" not in data["text"]
    
    # 3. Citation should be character-precise
    assert len(data["citations"]) == 1
    citation = data["citations"][0]
    assert citation["page"] == 1
    
    # Check that char_span is character-precise (covers only the cited words, not the whole chunk)
    char_start, char_end = citation["char_span"]
    chunks = load_chunks_from_db(doc_id)
    chunk_text = chunks[0].text
    cited_text = chunk_text[char_start:char_end]
    assert cited_text == "The total amount due is $450.00."
    
    # Check that bbox is horizontally interpolated
    # The original chunk bbox spans x0=0.0 to x1=1.0. 
    # Since the quote is in the middle of the chunk text, the sub-bbox x0 should be > 0.0 and x1 should be < 1.0.
    bbox = citation["bbox"]
    assert 0.0 < bbox["x0"] < bbox["x1"] < 1.0

def test_front_cascade_fabrication_blocking(tmp_path):
    test_file = tmp_path / "front_block.txt"
    test_file.write_text("The contract governing law is the laws of New York.")
    
    # Ingest
    res = client.post("/index", json={"path": str(test_file)})
    assert res.status_code == 200
    doc_id = res.json()["doc_id"]
    
    # Query with no overlap (fabricated)
    query = "What is the weather in Paris?"
    res_ask = client.post("/ask", json={"doc_id": doc_id, "query": query})
    assert res_ask.status_code == 200
    data = res_ask.json()
    
    # Should be blocked
    assert data["grounded"] is False
    assert data["text"] == "blocked"
    assert data["citations"] == []

def test_golden_set_precision():
    import json
    import pathlib
    
    base_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
    questions_file = base_dir / "bench" / "questions.json"
    
    assert questions_file.exists(), f"questions.json not found at {questions_file}"
    
    with open(questions_file, "r", encoding="utf-8") as f:
        fixtures_data = json.load(f)
        
    def get_file_path(filename):
        if filename == "unanswerable.pdf":
            return str(base_dir / "fixtures" / "unanswerable.pdf")
        elif filename.startswith("adversarial/"):
            rel = filename[len("adversarial/"):]
            return str(base_dir / "fixtures" / "adversarial" / rel)
        else:
            return str(base_dir / "fixtures" / "golden" / filename)
            
    grounded_count = 0
    precise_count = 0
    
    for fixture in fixtures_data:
        filename = fixture["filename"]
        filepath = get_file_path(filename)
        if not os.path.exists(filepath):
            continue
            
        # Index document
        res_idx = client.post("/index", json={"path": filepath})
        assert res_idx.status_code == 200
        doc_id = res_idx.json()["doc_id"]
        
        for q in fixture["questions"]:
            if not q["answerable"]:
                continue
                
            query = q["query"]
            res_ask = client.post("/ask", json={"doc_id": doc_id, "query": query})
            assert res_ask.status_code == 200
            data = res_ask.json()
            
            if data["grounded"]:
                grounded_count += 1
                
                # Check citations
                assert len(data["citations"]) > 0, f"Grounded answer for query '{query}' has no citations"
                
                # Get the chunk to compare length
                chunks = load_chunks_from_db(doc_id)
                
                # Check if all citations are character-precise
                is_all_precise = True
                for citation in data["citations"]:
                    # Find the corresponding chunk
                    chunk = next((c for c in chunks if c.chunk_id == citation["chunk_id"]), None)
                    if chunk:
                        char_start, char_end = citation["char_span"]
                        span_len = char_end - char_start
                        chunk_len = len(chunk.text)
                        
                        # A span is precise if it covers only the cited words, i.e., less than the whole chunk,
                        # OR if the chunk itself contains only one sentence (so it cannot be split further).
                        if span_len == chunk_len and chunk_len > 50:
                            parts = re.split(r"(?<=[.!?;])\s+|\n+", chunk.text.strip())
                            sentence_parts = [p.strip() for p in parts if len(p.strip()) >= 10]
                            if len(sentence_parts) > 1:
                                is_all_precise = False
                            
                if is_all_precise:
                    precise_count += 1
                    
    print(f"Grounded answers: {grounded_count}, Precise citations: {precise_count}")
    if grounded_count > 0:
        precision_pct = (precise_count / grounded_count) * 100.0
        print(f"Precision percentage: {precision_pct:.2f}%")
        assert precision_pct >= 95.0, f"Precision is {precision_pct:.2f}%, expected >= 95%"

