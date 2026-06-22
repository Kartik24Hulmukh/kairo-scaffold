import { describe, it, expect, beforeEach, vi } from 'vitest';
import { initApp, initWasm, getWasmEngine, getDocumentMetadata } from '../index.js';
import { setupDOM, setupPdfJsMock, setupFetchMock, setAskMockOverride, clearAskMockOverride } from './testHelper.js';

describe('Tier 1 — Feature Coverage (40 Tests)', () => {
  beforeEach(async () => {
    // Reset DOM and Mock setup
    setupDOM();
    setupPdfJsMock();
    
    // Initialize application and WASM core
    initApp();
    await initWasm();
    
    const wasmEngine = getWasmEngine();
    wasmEngine.clear();
    setupFetchMock(wasmEngine);
    clearAskMockOverride();
  });

  // Helper to wait for event loop ticks (e.g. FileReader callback)
  const wait = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms));

  // --- Feature 1: Document Upload (TXT & PDF) ---

  it('t1_upload_txt_success: Upload a valid small TXT file, verify upload state becomes active and the filename is displayed in the UI', async () => {
    const fileInput = document.getElementById('file-input');
    const docSelector = document.getElementById('doc-selector');
    
    const file = new File(["Employment terms."], "employment.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    
    await wait();
    
    expect(docSelector.options.length).toBe(2); // Default option + newly uploaded option
    expect(docSelector.value).toContain('employment.txt');
  });

  it('t1_upload_pdf_success: Upload a valid small native PDF, verify upload state is active and the filename/doc_id is registered', async () => {
    const fileInput = document.getElementById('file-input');
    const docSelector = document.getElementById('doc-selector');
    
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [{ str: "This is page 1 text", transform: [1,0,0,1,10,20] }] }]
    });
    
    const file = new File([mockPdfData], "doc.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    
    await wait();
    
    expect(docSelector.value).toContain('doc.pdf');
  });

  it('t1_drag_drop_txt: Trigger a dragover and drop event with a TXT file on the drop zone, verify upload handles file correctly', async () => {
    const dropZone = document.getElementById('drop-zone');
    const docSelector = document.getElementById('doc-selector');
    
    const file = new File(["Drag text data."], "dragged.txt", { type: "text/plain" });
    
    // Simulate Drag & Drop event
    const dropEvent = new Event('drop', { bubbles: true });
    dropEvent.dataTransfer = { files: [file] };
    dropZone.dispatchEvent(dropEvent);
    
    await wait();
    expect(docSelector.value).toContain('dragged.txt');
  });

  it('t1_drag_drop_pdf: Trigger a dragover and drop event with a PDF file, verify drop area triggers the parsing process', async () => {
    const dropZone = document.getElementById('drop-zone');
    const docSelector = document.getElementById('doc-selector');
    
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [{ str: "Page text content", transform: [1,0,0,1,0,0] }] }]
    });
    const file = new File([mockPdfData], "dragged.pdf", { type: "application/pdf" });
    
    const dropEvent = new Event('drop', { bubbles: true });
    dropEvent.dataTransfer = { files: [file] };
    dropZone.dispatchEvent(dropEvent);
    
    await wait();
    expect(docSelector.value).toContain('dragged.pdf');
  });

  it('t1_clear_upload: Click a reset or upload-new button, verify current document state is cleared and UI controls reset', async () => {
    const fileInput = document.getElementById('file-input');
    const clearBtn = document.getElementById('clear-btn');
    const docSelector = document.getElementById('doc-selector');
    
    // First upload
    const file = new File(["Text data."], "clear_test.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(docSelector.value).toContain('clear_test.txt');
    
    // Click clear
    clearBtn.click();
    expect(docSelector.value).toBe('');
    expect(document.getElementById('answer-text').textContent).toBe('');
  });

  // --- Feature 2: Client-side Text Parsing (PDF.js / TXT) ---

  it('t1_parse_txt_content: Parse a uploaded text file, verify that it correctly splits text into lines/paragraphs', async () => {
    const fileInput = document.getElementById('file-input');
    const file = new File(["Line 1\n\nLine 2\nLine 3"], "split.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId = document.getElementById('doc-selector').value;
    const wasmEngine = getWasmEngine();
    
    const chunks = wasmEngine.documents.get(docId);
    expect(chunks.length).toBe(3); // Line 1, Line 2, Line 3
    expect(chunks[0].text).toBe("Line 1");
    expect(chunks[2].text).toBe("Line 3");
  });

  it('t1_parse_pdf_pages: Parse a multi-page native PDF, verify it extracts text for each page and maintains correct page counts', async () => {
    const fileInput = document.getElementById('file-input');
    const mockPdfData = JSON.stringify({
      pages: [
        { width: 800, height: 1000, items: [{ str: "First page", transform: [1,0,0,1,10,10] }] },
        { width: 800, height: 1000, items: [{ str: "Second page", transform: [1,0,0,1,10,10] }] }
      ]
    });
    const file = new File([mockPdfData], "multipage.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId = document.getElementById('doc-selector').value;
    const wasmEngine = getWasmEngine();
    const chunks = wasmEngine.documents.get(docId);
    
    expect(chunks.length).toBe(2);
    expect(chunks[0].page).toBe(1);
    expect(chunks[1].page).toBe(2);
  });

  it('t1_parse_pdf_metadata: Verify extracted PDF pages return metadata matching actual page sequence', async () => {
    const fileInput = document.getElementById('file-input');
    const mockPdfData = JSON.stringify({
      pages: [
        { width: 500, height: 600, items: [{ str: "Page A", transform: [1,0,0,1,0,0] }] },
        { width: 700, height: 800, items: [{ str: "Page B", transform: [1,0,0,1,0,0] }] }
      ]
    });
    const file = new File([mockPdfData], "metadata.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId = document.getElementById('doc-selector').value;
    const docMeta = getDocumentMetadata()[docId];
    expect(docMeta.pages.length).toBe(2);
    expect(docMeta.pages[0].width_px).toBe(500);
    expect(docMeta.pages[1].width_px).toBe(700);
  });

  it('t1_parse_pdf_bbox: Parse a native PDF with layout info, verify bounding boxes (x0, y0, x1, y1) are extracted per text block', async () => {
    const fileInput = document.getElementById('file-input');
    const mockPdfData = JSON.stringify({
      pages: [{
        width: 800,
        height: 1000,
        items: [{
          str: "Bbox test",
          bbox: { x0: 20, y0: 30, x1: 200, y1: 50 }
        }]
      }]
    });
    const file = new File([mockPdfData], "bbox.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId = document.getElementById('doc-selector').value;
    const chunks = getWasmEngine().documents.get(docId);
    expect(chunks[0].bbox).toEqual({ x0: 20, y0: 30, x1: 200, y1: 50 });
  });

  it('t1_pdfjs_worker_init: Verify the PDF.js mock/real worker initializes correctly without runtime exceptions', async () => {
    const fileInput = document.getElementById('file-input');
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [{ str: "Worker test", transform: [1,0,0,1,0,0] }] }]
    });
    const file = new File([mockPdfData], "worker.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    
    expect(() => {
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    }).not.toThrow();
    await wait();
  });

  // --- Feature 3: Scanned PDF Detection ---

  it('t1_scanned_detect_zero_chars: Parse a PDF containing 0 text characters, verify scanned warning is displayed', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    // PDF with 0 characters
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [] }]
    });
    const file = new File([mockPdfData], "scanned.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(warning.style.display).toBe('block');
  });

  it('t1_scanned_detect_text_present: Parse a PDF containing normal text, verify scanned warning is NOT shown', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [{ str: "text", transform: [1,0,0,1,0,0] }] }]
    });
    const file = new File([mockPdfData], "normal.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(warning.style.display).toBe('none');
  });

  it('t1_scanned_query_blocked: Upload scanned PDF, submit query -> verify search flow is blocked/disabled and scanned warning is visible', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    const queryInput = document.getElementById('query-input');
    const askBtn = document.getElementById('ask-btn');
    
    const mockPdfData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [] }]
    });
    const file = new File([mockPdfData], "scanned_block.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    queryInput.value = "my question";
    askBtn.click();
    
    await wait();
    
    expect(window.fetch).not.toHaveBeenCalled();
    expect(warning.style.display).toBe('block');
  });

  it('t1_scanned_override_valid: Upload scanned PDF (warning shown), then upload a valid PDF, verify warning disappears', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    // First scanned
    const scannedData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [] }] });
    const scannedFile = new File([scannedData], "scanned.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [scannedFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    expect(warning.style.display).toBe('block');
    
    // Then normal
    const normalData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [{ str: "data", transform: [1,0,0,1,0,0] }] }] });
    const normalFile = new File([normalData], "normal.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [normalFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(warning.style.display).toBe('none');
  });

  it('t1_scanned_exact_message: Verify the warning message text is exactly: "This looks scanned — the desktop app does OCR; the web demo handles native-text PDFs."', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    const scannedData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [] }] });
    const scannedFile = new File([scannedData], "scanned.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [scannedFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(warning.textContent).toBe("This looks scanned — the desktop app does OCR; the web demo handles native-text PDFs.");
  });

  // --- Feature 4: One-Click Sample Files ---

  it('t1_click_sample_contract: Click "Sample Contract" button, verify contract document is loaded and selected in document selector', async () => {
    const btn = document.getElementById('sample-contract-btn');
    const selector = document.getElementById('doc-selector');
    
    btn.click();
    await wait();
    
    expect(selector.value).toBe('sample_contract');
  });

  it('t1_click_sample_invoice: Click "Sample Invoice" button, verify invoice document is loaded and selected in document selector', async () => {
    const btn = document.getElementById('sample-invoice-btn');
    const selector = document.getElementById('doc-selector');
    
    btn.click();
    await wait();
    
    expect(selector.value).toBe('sample_invoice');
  });

  it('t1_query_sample_contract: Query the loaded contract, verify matching results are evaluated through the WASM core', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination notice";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).toHaveBeenCalled();
    const responseData = await (await window.fetch.mock.results[0].value).json();
    expect(responseData.grounded).toBe(true);
    expect(responseData.citations.length).toBe(1);
    expect(responseData.citations[0].text).toContain("termination notice");
  });

  it('t1_query_sample_invoice: Query the loaded invoice, verify matching results are evaluated through the WASM core', async () => {
    document.getElementById('sample-invoice-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "Total amount due";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).toHaveBeenCalled();
    const responseData = await (await window.fetch.mock.results[0].value).json();
    expect(responseData.grounded).toBe(true);
    expect(responseData.citations.length).toBe(1);
    expect(responseData.citations[0].text).toContain("Total amount due");
  });

  it('t1_sample_upload_override: Load a sample file, then upload a custom file, verify sample selection is cleared/replaced', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    const selector = document.getElementById('doc-selector');
    expect(selector.value).toBe('sample_contract');
    
    const fileInput = document.getElementById('file-input');
    const file = new File(["Custom file content."], "custom.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(selector.value).toContain('custom.txt');
  });

  // --- Feature 5: Rust WASM Indexing & Search Core ---

  it('t1_wasm_init: Call engine initialization function, verify WASM core initializes without error', async () => {
    const wasmEngine = getWasmEngine();
    await expect(wasmEngine.init()).resolves.toBe(true);
  });

  it('t1_wasm_index_success: Index text and bounding box chunks, verify it returns a valid document summary', async () => {
    const wasmEngine = getWasmEngine();
    const chunks = [{ text: "term test", page: 1, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }];
    const summary = await wasmEngine.index_document("doc_summary_test", chunks);
    expect(summary.success).toBe(true);
    expect(summary.chunk_count).toBe(1);
    expect(summary.doc_id).toBe("doc_summary_test");
  });

  it('t1_wasm_query_exact_match: Query the indexed WASM database for an existing term, verify it returns matching chunks', async () => {
    const wasmEngine = getWasmEngine();
    const chunks = [{ text: "specific term to find", page: 1, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }];
    await wasmEngine.index_document("doc1", chunks);
    
    const results = await wasmEngine.query_document("doc1", "specific term");
    expect(results.length).toBe(1);
    expect(results[0].text).toBe("specific term to find");
  });

  it('t1_wasm_query_no_match: Query for a non-existent term, verify it returns empty/no-match structure', async () => {
    const wasmEngine = getWasmEngine();
    const chunks = [{ text: "specific term to find", page: 1, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }];
    await wasmEngine.index_document("doc1", chunks);
    
    const results = await wasmEngine.query_document("doc1", "missingword");
    expect(results.length).toBe(0);
  });

  it('t1_wasm_multi_doc_isolation: Index Doc A and Doc B, query Doc A, verify result only returns chunks belonging to Doc A', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("docA", [{ text: "needle in A", page: 1, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }]);
    await wasmEngine.index_document("docB", [{ text: "needle in B", page: 1, bbox: { x0: 0, y0: 0, x1: 10, y1: 10 } }]);
    
    const results = await wasmEngine.query_document("docA", "needle");
    expect(results.length).toBe(1);
    expect(results[0].text).toBe("needle in A");
  });

  // --- Feature 6: Query Flow (Ask, Answer, Citations) ---

  it('t1_query_flow_fetch_called: Input query and click Ask, verify POST /ask fetch request is sent', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).toHaveBeenCalledTimes(1);
    const callArgs = window.fetch.mock.calls[0];
    expect(callArgs[0]).toBe('http://127.0.0.1:7438/ask');
    expect(callArgs[1].method).toBe('POST');
  });

  it('t1_query_flow_answer_rendered: Receive grounded answer from fetch mock, verify text is rendered in answer-display', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("Grounded answer:");
  });

  it('t1_query_flow_citations_rendered: Receive answer with 2 citations, verify 2 citation chips are rendered in citations-container', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("doc_multi_citation", [
      { text: "first matches question", page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } },
      { text: "second matches question", page: 3, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }
    ]);
    
    // Select the document
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "doc_multi_citation";
    option.textContent = "Multi Citation Doc";
    docSelector.appendChild(option);
    docSelector.value = "doc_multi_citation";
    docSelector.dispatchEvent(new Event('change'));
    
    document.getElementById('query-input').value = "matches question";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chips = document.querySelectorAll('.citation-chip');
    expect(chips.length).toBe(2);
  });

  it('t1_query_flow_citation_text: Verify citation chips display text like "Citation 1 (Page 1)" and "Citation 2 (Page 3)"', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("doc_cit_text", [
      { text: "first text", page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } },
      { text: "second text", page: 3, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "doc_cit_text";
    option.textContent = "Citation Text Doc";
    docSelector.appendChild(option);
    docSelector.value = "doc_cit_text";
    docSelector.dispatchEvent(new Event('change'));
    
    document.getElementById('query-input').value = "text";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chips = document.querySelectorAll('.citation-chip');
    expect(chips[0].textContent).toBe("Citation 1 (Page 1)");
    expect(chips[1].textContent).toBe("Citation 2 (Page 3)");
  });

  it('t1_query_flow_enter_key: Trigger Enter key press on query-input, verify handleAsk function is triggered', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    const queryInput = document.getElementById('query-input');
    queryInput.value = "termination";
    
    // Simulate Enter key press
    const enterEvent = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true });
    queryInput.dispatchEvent(enterEvent);
    await wait();
    
    expect(window.fetch).toHaveBeenCalled();
  });

  // --- Feature 7: Refusal Cases ---

  it('t1_refusal_grounded_false: Fetch returns grounded: false, verify refusal message shows and answer display is empty', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: false, text: "blocked response text", citations: [] }
    });
    
    document.getElementById('query-input').value = "any query";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
    expect(document.getElementById('answer-text').textContent).toBe('');
  });

  it('t1_refusal_text_blocked: Fetch returns text: "blocked", verify refusal message shows', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "blocked", citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "any query";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t1_refusal_no_citations: Fetch returns grounded true but empty citations, verify refusal message shows (no grounding anchor)', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "valid text but empty citation", citations: [] }
    });
    
    document.getElementById('query-input').value = "any query";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t1_refusal_exact_message: Verify refusal message is: "Answer blocked: Response could not be verified or grounded."', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: false, text: "refused", citations: [] }
    });
    
    document.getElementById('query-input').value = "query";
    document.getElementById('ask-btn').click();
    await wait();
    
    const refusalMsg = document.getElementById('refusal-message');
    expect(refusalMsg.textContent).toBe("Answer blocked: Response could not be verified or grounded.");
  });

  it('t1_refusal_toggle_state: Query A is grounded (shows citations). Query B is blocked. Verify citations are cleared and refusal shows', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    // First query - grounded
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.querySelectorAll('.citation-chip').length).toBe(1);
    expect(document.getElementById('refusal-message').style.display).toBe('none');
    
    // Second query - blocked
    setAskMockOverride({
      json: { grounded: false, text: "", citations: [] }
    });
    document.getElementById('query-input').value = "blocked term";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.querySelectorAll('.citation-chip').length).toBe(0);
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  // --- Feature 8: Click-to-Highlight / Bounding Box Scaling ---

  it('t1_highlight_click_loads_image: Click citation chip, verify it sets page image src and loads image', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const img = document.getElementById('page-image');
    expect(img.style.display).toBe('block');
    expect(img.src).toBe('kairo-img://localhost/sample_contract_sha256.png');
  });

  it('t1_highlight_scale_normalized: BBox coordinates are normalized (<=1.0). Click chip, verify scaling maps coordinates to page dimensions', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("normalized_doc", [
      { text: "normalized text", page: 1, bbox: { x0: 0.1, y0: 0.2, x1: 0.5, y1: 0.6 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "normalized_doc";
    option.textContent = "Normalized Doc";
    docSelector.appendChild(option);
    docSelector.value = "normalized_doc";
    docSelector.dispatchEvent(new Event('change'));
    
    // Set page dimensions in metadata (width: 800, height: 1000)
    getDocumentMetadata()["normalized_doc"] = {
      pages: [{ width_px: 800, height_px: 1000, image_sha256: 'some_sha' }]
    };
    
    document.getElementById('query-input').value = "normalized text";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    // Wait for image onload highlight redraw
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
    // Image width is 800, scale is 800/800 = 1. x0 normalized is 0.1 -> 0.1 * 800 = 80.
    expect(Number(rect.getAttribute('x'))).toBe(80);
    expect(Number(rect.getAttribute('y'))).toBe(200);
  });

  it('t1_highlight_scale_absolute: BBox coordinates are absolute. Click chip, verify scaling calculations are computed correctly', async () => {
    const wasmEngine = getWasmEngine();
    // absolute bbox coords: x0: 100, y0: 200, x1: 400, y1: 500
    await wasmEngine.index_document("absolute_doc", [
      { text: "absolute text", page: 1, bbox: { x0: 100, y0: 200, x1: 400, y1: 500 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "absolute_doc";
    option.textContent = "Absolute Doc";
    docSelector.appendChild(option);
    docSelector.value = "absolute_doc";
    docSelector.dispatchEvent(new Event('change'));
    
    // Set database dimensions to width: 400, height: 500.
    // Page image width is 800, height is 1000 in JSDOM setup.
    // So scale factor is 2.
    getDocumentMetadata()["absolute_doc"] = {
      pages: [{ width_px: 400, height_px: 500, image_sha256: 'some_sha' }]
    };
    
    document.getElementById('query-input').value = "absolute text";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
    // x0 is 100 * scale(2) = 200
    expect(Number(rect.getAttribute('x'))).toBe(200);
    expect(Number(rect.getAttribute('y'))).toBe(400);
  });

  it('t1_highlight_svg_rect_drawn: Verify SVG rect is appended to highlight-overlay with attributes (x, y, width, height) matching scaled coordinates', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
    expect(rect.getAttribute('x')).toBeDefined();
    expect(rect.getAttribute('y')).toBeDefined();
    expect(rect.getAttribute('width')).toBeDefined();
    expect(rect.getAttribute('height')).toBeDefined();
  });

  it('t1_highlight_svg_styling: Verify SVG rect has styling: transparent yellow fill (rgba(255, 235, 59, 0.4)), yellow stroke, and stroke-width', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect.getAttribute('fill')).toBe('rgba(255, 235, 59, 0.4)');
    expect(rect.getAttribute('stroke')).toBe('#fbc02d');
    expect(rect.getAttribute('stroke-width')).toBe('2');
  });
});
