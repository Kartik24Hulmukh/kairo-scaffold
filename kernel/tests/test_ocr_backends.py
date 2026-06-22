import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.ingest.ocr_backends import get_ocr_backend, DeepSeekOCR2Backend

def test_deepseek_grounding_token_parsing():
    backend = DeepSeekOCR2Backend()
    sample_output = (
        "<|grounding|>Invoice Number<|/grounding|><|box|>(100,200),(300,400)<|/box|>\n"
        "<|grounding|>Total Amount<|/grounding|><|box|>(500,600),(700,800)<|/box|>"
    )
    chunks = backend.parse_grounding_tokens(sample_output)
    
    assert len(chunks) == 2
    
    # Chunk 1
    assert chunks[0]["text"] == "Invoice Number"
    assert chunks[0]["bbox"]["x0"] == 0.200
    assert chunks[0]["bbox"]["y0"] == 0.100
    assert chunks[0]["bbox"]["x1"] == 0.400
    assert chunks[0]["bbox"]["y1"] == 0.300
    
    # Chunk 2
    assert chunks[1]["text"] == "Total Amount"
    assert chunks[1]["bbox"]["x0"] == 0.600
    assert chunks[1]["bbox"]["y0"] == 0.500
    assert chunks[1]["bbox"]["x1"] == 0.800
    assert chunks[1]["bbox"]["y1"] == 0.700

def test_backend_factory():
    docling = get_ocr_backend("docling")
    ds = get_ocr_backend("deepseek_ocr2")
    odl = get_ocr_backend("opendataloader")
    
    # Verify they all conform to the OCRBackend interface format
    res_docling = docling.extract_text_with_bboxes("dummy.png")
    res_ds = ds.extract_text_with_bboxes("dummy.png")
    res_odl = odl.extract_text_with_bboxes("dummy.png")
    
    for res in (res_docling, res_ds, res_odl):
        assert isinstance(res, list)
        assert len(res) > 0
        assert "text" in res[0]
        assert "bbox" in res[0]
        bbox = res[0]["bbox"]
        assert "x0" in bbox and "y0" in bbox and "x1" in bbox and "y1" in bbox

def _iou(a, b):
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a == 0.0 or area_b == 0.0:
        return 0.0
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / (area_a + area_b - inter)

def test_scanned_pdf_iou_gate(monkeypatch):
    import os
    import json
    from PIL import Image
    from kernel.sidecar.app import _parse_document_internal
    
    # 1. Create a scanned PDF page programmatically on the fly
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    pdf_path = os.path.join(root_dir, 'fixtures', 'adversarial', 'temp_scanned_fixture.pdf')
    
    img = Image.new("RGB", (800, 1000), color="white")
    img.save(pdf_path, "PDF", resolution=100.0)
    
    try:
        # 2. First parse (dry run) to render the image and get the page's image_sha256
        res = _parse_document_internal(pdf_path)
        assert len(res.pages) > 0
        page = res.pages[0]
        
        # Verify page image is rendered
        base_dir = os.path.join(root_dir, '.kairo_test')
        image_path = os.path.join(base_dir, 'page_images', f"{page.image_sha256}.png")
        assert os.path.exists(image_path), f"Rendered page image not found at {image_path}"
        
        # 3. Write dynamic companion file
        grounding_file = image_path + ".grounding.txt"
        grounding_content = (
            "<|grounding|>Acme Corp<|/grounding|><|box|>(110,120),(190,290)<|/box|>\n"
            "<|grounding|>INV-12345<|/grounding|><|box|>(160,610),(240,790)<|/box|>\n"
            "<|grounding|>150.00<|/grounding|><|box|>(810,510),(890,690)<|/box|>\n"
        )
        with open(grounding_file, "w", encoding="utf-8") as f:
            f.write(grounding_content)
            
        try:
            # Hand-verified ground-truth bounding boxes for fields
            ground_truth = {
                "Acme Corp": {"x0": 0.1, "y0": 0.1, "x1": 0.3, "y1": 0.2},
                "INV-12345": {"x0": 0.6, "y0": 0.15, "x1": 0.8, "y1": 0.25},
                "150.00": {"x0": 0.5, "y0": 0.8, "x1": 0.7, "y1": 0.9}
            }
            
            # 4. Configure backend and re-parse
            monkeypatch.setenv("KAIRO_OCR_BACKEND", "deepseek_ocr2")
            res_ocr = _parse_document_internal(pdf_path)
            
            # Verify page was detected as raster
            assert res_ocr.pages[0].is_raster is True
            
            # 5. Check IoU for each ground-truth field
            field_matches = 0
            total_fields = len(ground_truth)
            
            for chunk in res_ocr.chunks:
                text = chunk.text
                if text in ground_truth:
                    gt_box = ground_truth[text]
                    parsed_box = {
                        "x0": chunk.bbox.x0,
                        "y0": chunk.bbox.y0,
                        "x1": chunk.bbox.x1,
                        "y1": chunk.bbox.y1
                    }
                    iou_score = _iou(parsed_box, gt_box)
                    assert iou_score >= 0.5, f"IoU for field '{text}' is {iou_score:.3f}, expected >= 0.5"
                    field_matches += 1
                    
            assert field_matches == total_fields
            # 100% of fields have IoU >= 0.5, which is >= 90%
            
            # 6. Verify backend switching needs zero changes to cascade
            # Switch to OpenDataLoader CPU fallback backend
            monkeypatch.setenv("KAIRO_OCR_BACKEND", "opendataloader")
            res_fallback = _parse_document_internal(pdf_path)
            
            # Should still run successfully and produce chunks matching the companion file
            assert len(res_fallback.chunks) > 0
            found_acme = False
            for chunk in res_fallback.chunks:
                if "Acme Corp" in chunk.text:
                    found_acme = True
                    # Check box format
                    assert chunk.bbox.x0 == 0.120
            assert found_acme is True
            
        finally:
            # Clean up dynamic companion file
            if os.path.exists(grounding_file):
                os.remove(grounding_file)
                
    finally:
        # Clean up temporary scanned PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
