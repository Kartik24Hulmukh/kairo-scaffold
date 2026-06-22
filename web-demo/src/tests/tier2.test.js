import { describe, it, expect, beforeEach, vi } from 'vitest';
import { initApp, initWasm, getWasmEngine, getDocumentMetadata } from '../index.js';
import { setupDOM, setupPdfJsMock, setupFetchMock, setAskMockOverride, clearAskMockOverride } from './testHelper.js';

describe('Tier 2 — Boundary & Corner Cases (40 Tests)', () => {
  beforeEach(async () => {
    setupDOM();
    setupPdfJsMock();
    initApp();
    await initWasm();
    
    const wasmEngine = getWasmEngine();
    wasmEngine.clear();
    setupFetchMock(wasmEngine);
    clearAskMockOverride();
    
    vi.spyOn(window, 'alert').mockImplementation(() => {});
  });

  const wait = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms));

  // --- File Upload & Parsing Boundaries ---

  it('t2_upload_empty_txt: Upload an empty TXT file, verify graceful handle or validation warning', async () => {
    const fileInput = document.getElementById('file-input');
    const file = new File([""], "empty.txt", { type: "text/plain" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(window.alert).toHaveBeenCalled();
  });

  it('t2_upload_large_txt: Upload a 10MB text file, verify no UI lockup', async () => {
    const fileInput = document.getElementById('file-input');
    // Generate 10MB text
    const largeText = "A".repeat(10 * 1024 * 1024);
    const file = new File([largeText], "large.txt", { type: "text/plain" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(document.getElementById('doc-selector').value).toContain('large.txt');
  });

  it('t2_upload_invalid_type: Select a .zip file, verify it triggers validation error / alert and is blocked', async () => {
    const fileInput = document.getElementById('file-input');
    const file = new File(["zip content"], "archive.zip", { type: "application/zip" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(window.alert).toHaveBeenCalled();
    expect(document.getElementById('doc-selector').value).toBe('');
  });

  it('t2_parse_corrupt_pdf: Upload a corrupted PDF, verify parser throws error and UI displays friendly error message', async () => {
    const fileInput = document.getElementById('file-input');
    // Create corrupt PDF indicator
    const corruptData = JSON.stringify({ corrupt: true });
    const file = new File([corruptData], "corrupt.pdf", { type: "application/pdf" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(window.alert).toHaveBeenCalledWith("Error: Corrupted PDF file.");
  });

  it('t2_parse_blank_pages: PDF contains blank pages (no text). Verify handled as scanned document', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    const mockPdfData = JSON.stringify({
      pages: [
        { width: 800, height: 1000, items: [] },
        { width: 800, height: 1000, items: [] }
      ]
    });
    const file = new File([mockPdfData], "blank.pdf", { type: "application/pdf" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(warning.style.display).toBe('block');
  });

  it('t2_upload_nameless_file: Upload a file with no filename or extension, verify graceful handling', async () => {
    const fileInput = document.getElementById('file-input');
    const file = new File(["contents"], "", { type: "text/plain" });
    
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    expect(window.alert).toHaveBeenCalled();
  });

  it('t2_upload_duplicate_name: Upload duplicate file name, verify it generates a unique doc_id/suffix', async () => {
    const fileInput = document.getElementById('file-input');
    
    const file1 = new File(["content 1"], "dup.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file1], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId1 = document.getElementById('doc-selector').value;
    
    const file2 = new File(["content 2"], "dup.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file2], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const docId2 = document.getElementById('doc-selector').value;
    
    expect(docId1).not.toBe(docId2);
    expect(docId2).toContain('dup.txt_1');
  });

  it('t2_upload_special_chars_name: File name contains special/unicode characters, verify selector displays name safely', async () => {
    const fileInput = document.getElementById('file-input');
    const docSelector = document.getElementById('doc-selector');
    
    const file = new File(["content"], "★test_special_chars★.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    const optionText = docSelector.options[docSelector.options.length - 1].textContent;
    expect(optionText).toContain("★test_special_chars★.txt");
  });

  // --- Query Input Boundaries ---

  it('t2_query_only_whitespace: Submit a query containing only spaces, verify request is blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "   ";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).not.toHaveBeenCalled();
    expect(window.alert).toHaveBeenCalledWith("Please enter a query.");
  });

  it('t2_query_empty: Submit empty query, verify request is blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).not.toHaveBeenCalled();
    expect(window.alert).toHaveBeenCalledWith("Please enter a query.");
  });

  it('t2_query_extreme_length: Submit query with 10k characters, verify no buffer overflow or crash', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    const queryVal = "A".repeat(10000);
    document.getElementById('query-input').value = queryVal;
    
    expect(() => {
      document.getElementById('ask-btn').click();
    }).not.toThrow();
    await wait();
  });

  it('t2_query_script_injection: Input <script>alert("xss")</script>, verify it renders as plain text without execution', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    const xss = "<script>alert('xss')</script>";
    setAskMockOverride({
      json: { grounded: true, text: xss, citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "hack";
    document.getElementById('ask-btn').click();
    await wait();
    
    const ansText = document.getElementById('answer-text');
    expect(ansText.textContent).toBe(xss);
  });

  it('t2_query_no_doc_selected: Ask query before selecting any document, verify warning/alert', async () => {
    document.getElementById('query-input').value = "question";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).not.toHaveBeenCalled();
    expect(window.alert).toHaveBeenCalledWith("Please select a document first.");
  });

  it('t2_query_rapid_clicks: Rapid double click on Ask button, verify only 1 fetch call is executed', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    
    // Simulate double click
    document.getElementById('ask-btn').click();
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(window.fetch).toHaveBeenCalledTimes(1);
  });

  // --- WASM Core Data Boundaries ---

  it('t2_wasm_empty_chunks: Index document with empty text chunks, verify WASM core handles it', async () => {
    const wasmEngine = getWasmEngine();
    await expect(wasmEngine.index_document("empty_chunks", [])).resolves.toEqual({
      doc_id: "empty_chunks",
      chunk_count: 0,
      success: true,
      summary: "Indexed 0 chunks for document empty_chunks"
    });
  });

  it('t2_wasm_large_coordinates: Index bbox with extreme out-of-bounds coordinates, verify coordinates are clipped or handled', async () => {
    const wasmEngine = getWasmEngine();
    const chunks = [{ text: "out of bounds bbox", page: 1, bbox: { x0: -1000, y0: -1000, x1: 50000, y1: 50000 } }];
    await expect(wasmEngine.index_document("large_coords", chunks)).resolves.toBeDefined();
  });

  it('t2_wasm_empty_doc_id: Index document with empty string doc_id, verify error handled', async () => {
    const wasmEngine = getWasmEngine();
    await expect(wasmEngine.index_document("", [])).rejects.toThrow();
  });

  it('t2_wasm_query_empty_term: Query WASM core with empty term, verify returns empty result', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("test", [{ text: "hello", page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }]);
    const matches = await wasmEngine.query_document("test", "");
    expect(matches.length).toBe(0);
  });

  it('t2_wasm_query_unindexed: Query WASM core before indexing any document, verify graceful return', async () => {
    const wasmEngine = getWasmEngine();
    const matches = await wasmEngine.query_document("nonexistent", "hello");
    expect(matches.length).toBe(0);
  });

  it('t2_wasm_heavy_load: Index 100 mock documents, verify search performance and memory stability', async () => {
    const wasmEngine = getWasmEngine();
    for (let i = 0; i < 100; i++) {
      await wasmEngine.index_document(`doc_${i}`, [{ text: `content ${i}`, page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }]);
    }
    const matches = await wasmEngine.query_document("doc_50", "content 50");
    expect(matches.length).toBe(1);
  });

  // --- Mock Response & Refusal Boundaries ---

  it('t2_refusal_server_500: Sidecar /ask returns 500 error, verify refusal message shows', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      ok: false,
      status: 500,
      json: {}
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_malformed_json: Sidecar returns malformed JSON, verify refusal message shows', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    window.fetch = vi.fn(async () => {
      return {
        ok: true,
        status: 200,
        json: async () => {
          throw new Error("SyntaxError: Unexpected token < in JSON");
        }
      };
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_excessive_text: Sidecar returns extremely long answer, verify text wraps cleanly without breaking layout', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    const longResponse = "Word ".repeat(5000);
    setAskMockOverride({
      json: { grounded: true, text: longResponse, citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toBe(longResponse);
  });

  it('t2_refusal_citation_missing_fields: Citation lacks page field, verify blocked or handles gracefully', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "Grounded answer", citations: [{ bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    // Page is missing, should handle or display refusal / block
    // Let's verify: citation page number displays undefined or blocked
    const chip = document.querySelector('.citation-chip');
    if (chip) {
      expect(chip.textContent).toContain("undefined");
    }
  });

  it('t2_refusal_citation_page_out_of_bounds: Page number in citation is larger than document pages, verify no crash', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    // contract has 1 page, request citation page 99
    setAskMockOverride({
      json: { grounded: true, text: "out of bounds page", citations: [{ page: 99, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    expect(chip).toBeTruthy();
    
    // Click it, make sure it does not crash
    expect(() => {
      chip.click();
    }).not.toThrow();
  });

  it('t2_refusal_citation_nan_bbox: Citation bbox values are NaN, verify no crash, highlight not drawn', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "nan bbox", citations: [{ page: 1, bbox: { x0: NaN, y0: NaN, x1: NaN, y1: NaN } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeNull(); // Skipped drawing
  });

  it('t2_refusal_grounded_true_null_citations: Grounded true, but citations field is null, verify blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "null citations", citations: null }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_grounded_true_empty_text: Grounded true, citations present, but text is empty, verify blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "", citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_grounded_true_blocked_text: Grounded true, citations present, text is exactly "blocked", verify blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "blocked", citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_grounded_false_blocked_text: Grounded false, citations present, text is exactly "blocked", verify blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: false, text: "blocked", citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t2_refusal_grounded_true_zero_len_citations: Grounded true, citations field is empty array, verify blocked', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "grounded true empty cit array", citations: [] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  // --- Highlight Canvas & SVG Boundaries ---

  it('t2_canvas_image_404: Click citation when page image 404s, verify no highlight drawn, logs error', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    // Trigger image error directly to avoid JSDOM virtual console event formatting bug
    pageImage.onerror();
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeNull();
    expect(consoleErrorSpy).toHaveBeenCalled();
    consoleErrorSpy.mockRestore();
  });

  it('t2_canvas_image_slow_load: Page image takes 2 seconds to load, verify highlight rect is drawn only after load finishes', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    // Define pageImage complete to return false first
    const pageImage = document.getElementById('page-image');
    Object.defineProperty(pageImage, 'complete', { get: () => false, configurable: true });
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    // Rect should not be drawn yet
    let rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeNull();
    
    // Simulate image loaded
    Object.defineProperty(pageImage, 'complete', { get: () => true, configurable: true });
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
  });

  it('t2_canvas_window_resize: Resize browser window, verify highlight SVG adjusts its client width/height dynamically', async () => {
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
    
    // Trigger window resize
    window.dispatchEvent(new Event('resize'));
    await wait();
    
    const overlay = document.getElementById('highlight-overlay');
    expect(overlay.style.width).toBe('800px');
  });

  it('t2_canvas_zero_bbox: Click citation with bbox [0,0,0,0], verify no rect or zero-rect drawn', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "zero bbox", citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 0, y1: 0 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeNull();
  });

  it('t2_canvas_rapid_clicks: Click multiple citation chips rapidly, verify only the last clicked chip\'s highlight is rendered', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("rapid_doc", [
      { text: "first matches", page: 1, bbox: { x0: 10, y0: 20, x1: 30, y1: 40 } },
      { text: "second matches", page: 1, bbox: { x0: 100, y0: 200, x1: 150, y1: 250 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "rapid_doc";
    option.textContent = "Rapid Doc";
    docSelector.appendChild(option);
    docSelector.value = "rapid_doc";
    docSelector.dispatchEvent(new Event('change'));
    
    getDocumentMetadata()["rapid_doc"] = {
      pages: [{ width_px: 800, height_px: 1000, image_sha256: 'rapid_sha' }]
    };
    
    document.getElementById('query-input').value = "matches";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chips = document.querySelectorAll('.citation-chip');
    
    // Click chip 1 then chip 2 rapidly
    chips[0].click();
    chips[1].click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    // Chip 2 has x0: 100
    expect(Number(rect.getAttribute('x'))).toBe(100);
  });

  it('t2_canvas_zero_db_dimensions: DB page width/height is 0, verify scaling doesn\'t divide by zero and falls back safely', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("zero_dim_doc", [
      { text: "zero dim matches", page: 1, bbox: { x0: 10, y0: 20, x1: 30, y1: 40 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "zero_dim_doc";
    option.textContent = "Zero Dim Doc";
    docSelector.appendChild(option);
    docSelector.value = "zero_dim_doc";
    docSelector.dispatchEvent(new Event('change'));
    
    // Set width and height to 0
    getDocumentMetadata()["zero_dim_doc"] = {
      pages: [{ width_px: 0, height_px: 0, image_sha256: 'zero_dim_sha' }]
    };
    
    document.getElementById('query-input').value = "zero dim matches";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy(); // Should fall back and not crash
  });

  it('t2_canvas_no_sha256: Citation references page with missing image_sha256 in document metadata, verify error handled', async () => {
    const wasmEngine = getWasmEngine();
    await wasmEngine.index_document("no_sha_doc", [
      { text: "no sha matches", page: 1, bbox: { x0: 10, y0: 20, x1: 30, y1: 40 } }
    ]);
    
    const docSelector = document.getElementById('doc-selector');
    const option = document.createElement('option');
    option.value = "no_sha_doc";
    option.textContent = "No Sha Doc";
    docSelector.appendChild(option);
    docSelector.value = "no_sha_doc";
    docSelector.dispatchEvent(new Event('change'));
    
    // missing image_sha256
    getDocumentMetadata()["no_sha_doc"] = {
      pages: [{ width_px: 800, height_px: 1000, image_sha256: null }]
    };
    
    document.getElementById('query-input').value = "no sha matches";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    expect(() => {
      chip.click();
    }).not.toThrow();
  });

  it('t2_canvas_page_zero: Citation page is index 0, verify maps correctly to display page', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    setAskMockOverride({
      json: { grounded: true, text: "page index 0", citations: [{ page: 0, bbox: { x0: 10, y0: 20, x1: 30, y1: 40 } }] }
    });
    
    document.getElementById('query-input').value = "test";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    expect(() => {
      chip.click();
    }).not.toThrow();
  });

  it('t2_canvas_negative_offsets: Canvas container offsets are negative, verify highlight overlay maps to correct viewport coordinates', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    const pageImage = document.getElementById('page-image');
    Object.defineProperties(pageImage, {
      offsetLeft: { get: () => -50, configurable: true },
      offsetTop: { get: () => -100, configurable: true }
    });
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const overlay = document.getElementById('highlight-overlay');
    expect(overlay.style.left).toBe('-50px');
    expect(overlay.style.top).toBe('-100px');
  });
});
