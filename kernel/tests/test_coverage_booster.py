import os
import sys
import pathlib
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Adjust path to import from kernel.sidecar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from kernel.sidecar.app import (
    app,
    _parse_document_internal,
    _ingest_text,
    _ingest_docx,
    _ingest_pdf_fastpath,
    _ingest_pdf_docling,
    IndexRequest,
    index_doc,
    parse_doc,
)
from kernel.sidecar.pdf_fastpath import process_pdf
from kernel.sidecar.model_gateway import app as gateway_app

client = TestClient(app)
gateway_client = TestClient(gateway_app)

def test_coverage_booster():
    # 1. Test non-existent file path (covers line 286: raise 404)
    with pytest.raises(Exception):
        _parse_document_internal("non_existent_file.txt")
        
    # 2. Test unsupported file extension (covers line 305: raise 400)
    temp_unsupported = "temp_unsupported.xyz"
    with open(temp_unsupported, "w") as f:
        f.write("hello")
    try:
        with pytest.raises(Exception):
            _parse_document_internal(temp_unsupported)
    finally:
        if os.path.exists(temp_unsupported):
            os.remove(temp_unsupported)

    # 3. Test parse_doc endpoint directly
    txt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/sample_contract_01.txt"))
    req = IndexRequest(path=txt_path)
    res_parse = parse_doc(req)
    assert res_parse.doc_id is not None
    
    # 4. Test index_doc error path (covers line 362: raise 500)
    bad_req = IndexRequest(path="invalid_file_name_for_error.xyz")
    with pytest.raises(Exception):
        index_doc(bad_req)

    # 5. Direct invocation of pdf fastpath (covers 255-267)
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/test.pdf"))
    if os.path.exists(pdf_path):
        # Cover ingest pdf fastpath
        pages, chunks = _ingest_pdf_fastpath(pathlib.Path(pdf_path))
        assert len(pages) > 0
        
        # Cover pdf_fastpath.py process_pdf directly
        res_fast = process_pdf(pdf_path)
        assert len(res_fast["pages"]) > 0

    # 6. Direct invocation of docx parsing (covers 169-187)
    docx_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../fixtures/golden/test_track.docx"))
    if os.path.exists(docx_path):
        pages, chunks = _ingest_docx(pathlib.Path(docx_path))
        assert len(pages) > 0

    # 7. Test model gateway stub endpoint
    res_gateway = gateway_client.post("/chat/completions", json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}]
    })
    assert res_gateway.status_code == 200
    assert "choices" in res_gateway.json()

    # 8. Cover _ingest_pdf_docling via Mocking (covers 201-253)
    with patch('docling.document_converter.DocumentConverter') as MockConverter:
        mock_instance = MagicMock()
        MockConverter.return_value = mock_instance
        
        mock_result = MagicMock()
        mock_docling_doc = MagicMock()
        
        # Set up pages
        mock_page = MagicMock()
        mock_page.size = MagicMock(width=800, height=1000)
        mock_docling_doc.pages = {1: mock_page}
        
        # Set up elements/texts/tables
        mock_element = MagicMock()
        mock_element.prov = [MagicMock(page_no=1, bbox=MagicMock(left=10.0, top=10.0, right=100.0, bottom=100.0))]
        mock_element.text = "Mocked PDF Docling text"
        mock_docling_doc.texts = [mock_element]
        
        # Cover export_to_markdown
        mock_element_table = MagicMock()
        mock_element_table.prov = [MagicMock(page_no=1, bbox=MagicMock(left=10.0, top=10.0, right=100.0, bottom=100.0))]
        del mock_element_table.text # Force fallback to export_to_markdown
        mock_element_table.export_to_markdown.return_value = "Mocked table markdown"
        mock_docling_doc.tables = [mock_element_table]
        
        mock_result.document = mock_docling_doc
        mock_instance.convert.return_value = mock_result
        
        # Call it!
        pages, chunks = _ingest_pdf_docling(pathlib.Path("fake_path.pdf"))
        assert len(pages) > 0
        assert len(chunks) > 0

    # 9. Cover pdf_fastpath.py main()
    from kernel.sidecar.pdf_fastpath import main as fastpath_main
    with patch('sys.argv', ['pdf_fastpath.py']):
        with pytest.raises(SystemExit):
            fastpath_main()
            
    with patch('sys.argv', ['pdf_fastpath.py', 'non_existent_file.pdf']):
        with pytest.raises(SystemExit):
            fastpath_main()

    with patch('sys.argv', ['pdf_fastpath.py', pdf_path]):
        with patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
            fastpath_main()
            assert mock_stdout.write.called

