import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from packs.generic.pack import GenericPack, ExtractionCandidate
from packs.invoice.pack import InvoicePack
from packs.paper.pack import PaperPack
from packs.contract.pack import ContractPack

def test_packs():
    p1 = GenericPack()
    assert "summary" in p1.fields
    assert p1.extract("doc_nonexistent") == []
    assert isinstance(p1.schema, dict)
    assert "summary" in p1.schema.get("properties", {})
    assert p1.prompt
    assert hasattr(p1, "examples")
    assert len(p1.examples) >= 1, "GenericPack must have at least one few-shot example"
    assert hasattr(p1, "validators")
    assert len(p1.validators) >= 1

    p2 = InvoicePack()
    assert "vendor_name" in p2.fields
    assert p2.extract("doc_nonexistent") == []
    assert isinstance(p2.schema, dict)
    assert "vendor_name" in p2.schema.get("properties", {})
    assert p2.prompt
    assert hasattr(p2, "examples")
    assert len(p2.examples) >= 1, "InvoicePack must have at least one few-shot example"
    assert hasattr(p2, "validators")
    assert len(p2.validators) >= 1

    p3 = PaperPack()
    assert "title" in p3.fields
    assert p3.extract("doc_nonexistent") == []
    assert isinstance(p3.schema, dict)
    assert "title" in p3.schema.get("properties", {})
    assert p3.prompt
    assert hasattr(p3, "examples")
    assert len(p3.examples) >= 1, "PaperPack must have at least one few-shot example"
    assert hasattr(p3, "validators")
    assert len(p3.validators) >= 1

    p4 = ContractPack()
    assert "parties" in p4.fields
    assert p4.extract("doc_nonexistent") == []
    assert isinstance(p4.schema, dict)
    assert "parties" in p4.schema.get("properties", {})
    assert p4.prompt
    assert hasattr(p4, "examples")
    assert len(p4.examples) >= 1, "ContractPack must have at least one few-shot example"
    assert hasattr(p4, "validators")
    assert len(p4.validators) >= 1

    # Test validate_grounding behavior
    for pack in [p1, p2, p3, p4]:
        validate_fn = pack.validators[0]
        
        # Scenario 1: Not found, not a default -> block/reject
        cand = ExtractionCandidate("vendor_name", "Acme", "Acme", 0.9, "c1")
        def mock_verify_block(val, doc_id, chunks):
            return "block", []
        assert validate_fn(cand, "doc_1", [], mock_verify_block) is False

        # Scenario 2: Found -> accept
        def mock_verify_found(val, doc_id, chunks):
            return "exact", [{"chunk_id": "c1", "char_span": (0, 4), "page": 1, "bbox": {"x0":0,"y0":0,"x1":1,"y1":1}}]
        assert validate_fn(cand, "doc_1", [], mock_verify_found) is True

        # Scenario 3: Not found but allowed default "tax_amount" = "0.00" -> bypass/accept
        cand_tax = ExtractionCandidate("tax_amount", "0.00", "0.00", 0.9, "c1")
        assert validate_fn(cand_tax, "doc_1", [], mock_verify_block) is True

        # Scenario 4: Not found but allowed default "payment_terms" = "Net 30" -> bypass/accept
        cand_terms = ExtractionCandidate("payment_terms", "Net 30", "Net 30", 0.9, "c1")
        assert validate_fn(cand_terms, "doc_1", [], mock_verify_block) is True

        # Scenario 5: Not found but allowed default "currency" = "USD" -> bypass/accept
        cand_curr = ExtractionCandidate("currency", "USD", "USD", 0.9, "c1")
        assert validate_fn(cand_curr, "doc_1", [], mock_verify_block) is True

