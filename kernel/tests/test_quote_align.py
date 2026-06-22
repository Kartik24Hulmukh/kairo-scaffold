import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.ingest.quote_align import smith_waterman_align, align_quote_to_chunks

class MockBBox:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

class MockChunk:
    def __init__(self, chunk_id, text, page, bbox):
        self.id = chunk_id
        self.text = text
        self.page = page
        self.bbox = bbox

def test_smith_waterman_exact():
    ratio, start, end = smith_waterman_align("Mock Vendor", "This invoice is from Mock Vendor Inc.")
    assert ratio == 1.0
    assert start == 21
    assert end == 32
    assert "This invoice is from Mock Vendor Inc."[start:end] == "Mock Vendor"

def test_smith_waterman_fuzzy():
    # Minor mismatch: 'o' -> '0'
    ratio, start, end = smith_waterman_align("M0ck Vendor", "This invoice is from Mock Vendor Inc.")
    assert ratio >= 0.85
    assert start == 21
    assert end == 32

def test_align_quote_to_chunks():
    chunks = [
        MockChunk("c0", "First paragraph of the document.", 1, MockBBox(0.0, 0.0, 1.0, 0.1)),
        MockChunk("c1", "Total Amount: $120.50 due on receipt.", 1, MockBBox(0.0, 0.1, 1.0, 0.2)),
    ]
    
    # Matching quote
    res = align_quote_to_chunks("Total Amount", chunks)
    assert res is not None
    assert res["chunk_id"] == "c1"
    assert res["page"] == 1
    assert res["char_span"] == (0, 12)
    # The sub bbox x-coords should start at 0.0 and cover some length
    assert res["bbox"]["x0"] == 0.0
    assert res["bbox"]["x1"] > 0.0
    
    # Non-matching quote
    res_fail = align_quote_to_chunks("Non existent company", chunks)
    assert res_fail is None
