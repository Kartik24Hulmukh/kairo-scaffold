import { describe, it, expect, beforeEach, vi } from 'vitest';
import { initApp, initWasm, getWasmEngine, getDocumentMetadata } from '../index.js';
import { setupDOM, setupPdfJsMock, setupFetchMock, setAskMockOverride, clearAskMockOverride } from './testHelper.js';

describe('Tier 3 — Cross-Feature Combinations (8 Tests)', () => {
  beforeEach(async () => {
    setupDOM();
    setupPdfJsMock();
    initApp();
    await initWasm();
    
    const wasmEngine = getWasmEngine();
    wasmEngine.clear();
    setupFetchMock(wasmEngine);
    clearAskMockOverride();
  });

  const wait = (ms = 20) => new Promise(resolve => setTimeout(resolve, ms));

  it('t3_upload_query_switch_query: Upload Doc A, ask question. Upload Doc B, ask question. Verify results are isolated', async () => {
    const fileInput = document.getElementById('file-input');
    const queryInput = document.getElementById('query-input');
    const askBtn = document.getElementById('ask-btn');
    const ansText = document.getElementById('answer-text');
    
    // Upload Doc A
    const fileA = new File(["apple fruit information"], "docA.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [fileA], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    queryInput.value = "apple";
    askBtn.click();
    await wait();
    expect(ansText.textContent).toContain("apple");
    
    // Upload Doc B
    const fileB = new File(["banana banana fruit"], "docB.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [fileB], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    // Query Doc B
    queryInput.value = "banana";
    askBtn.click();
    await wait();
    expect(ansText.textContent).toContain("banana");
    
    // Query "apple" on Doc B should fail (grounded: false)
    queryInput.value = "apple";
    askBtn.click();
    await wait();
    expect(document.getElementById('refusal-message').style.display).toBe('block');
  });

  it('t3_sample_upload_override_query: Select Sample Contract, query it. Then upload custom TXT file, query it. Verify search context switches', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("termination");
    
    // Upload custom txt
    const fileInput = document.getElementById('file-input');
    const file = new File(["some custom document content here"], "custom.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    // Selector should be switched to custom.txt
    expect(document.getElementById('doc-selector').value).toContain('custom.txt');
    
    // Querying "custom document" should work now
    document.getElementById('query-input').value = "custom document";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("custom document");
  });

  it('t3_scanned_warning_lifecycle: Upload scanned PDF (warning shown), then upload valid PDF (warning removed). Query valid PDF (shows answer)', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    
    // Upload scanned PDF
    const scannedData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [] }] });
    const scannedFile = new File([scannedData], "scanned.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [scannedFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    expect(warning.style.display).toBe('block');
    
    // Upload valid PDF
    const validData = JSON.stringify({
      pages: [{ width: 800, height: 1000, items: [{ str: "valid page text info", transform: [1,0,0,1,0,0] }] }]
    });
    const validFile = new File([validData], "valid.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [validFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    expect(warning.style.display).toBe('none');
    
    // Query it
    document.getElementById('query-input').value = "valid page text";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("valid page text");
  });

  it('t3_citation_clearance_on_refusal: Ask query (returns citations & answers). Ask second query (returns refusal). Verify old citations and highlight overlay are cleared', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.querySelectorAll('.citation-chip').length).toBe(1);
    
    // Set refusal override
    setAskMockOverride({
      json: { grounded: false, text: "", citations: [] }
    });
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.querySelectorAll('.citation-chip').length).toBe(0);
    expect(document.getElementById('highlight-overlay').innerHTML).toBe('');
    expect(document.getElementById('answer-text').textContent).toBe('');
  });

  it('t3_resize_during_highlight: Click citation (highlight drawn). Resize window. Verify highlight coordinates re-align to updated layout', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    document.querySelector('.citation-chip').click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const overlay = document.getElementById('highlight-overlay');
    expect(overlay.style.width).toBe('800px');
    
    // Change image mock width to 900
    Object.defineProperty(pageImage, 'clientWidth', { get: () => 900, configurable: true });
    
    // Resize
    window.dispatchEvent(new Event('resize'));
    await wait();
    
    expect(overlay.style.width).toBe('900px');
  });

  it('t3_selector_reset_clears_ui: Ask query (citations and answers shown). Switch document selector back to default empty option. Verify citations, answer text, and highlighted image are cleared', async () => {
    document.getElementById('sample-contract-btn').click();
    await wait();
    
    document.getElementById('query-input').value = "termination";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("termination");
    expect(document.querySelectorAll('.citation-chip').length).toBe(1);
    
    // Reset selector
    const selector = document.getElementById('doc-selector');
    selector.value = '';
    selector.dispatchEvent(new Event('change'));
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toBe('');
    expect(document.querySelectorAll('.citation-chip').length).toBe(0);
    expect(document.getElementById('page-image').style.display).toBe('none');
  });

  it('t3_upload_txt_no_highlights: Upload TXT file. Ask question (shows grounded answer and citation). Click citation, verify it handles TXT source view gracefully (no image viewer errors, display raw text if image doesn\'t exist)', async () => {
    const fileInput = document.getElementById('file-input');
    const file = new File(["legal term and definitions"], "terms.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    document.getElementById('query-input').value = "legal term";
    document.getElementById('ask-btn').click();
    await wait();
    
    const chip = document.querySelector('.citation-chip');
    expect(chip).toBeTruthy();
    
    // Clicking should handle gracefully (no image exists since sha256 is null)
    expect(() => {
      chip.click();
    }).not.toThrow();
    
    const pageImage = document.getElementById('page-image');
    expect(pageImage.style.display).toBe('none');
  });

  it('t3_multi_upload_wasm_persistence: Upload three different files. Query each file by selecting it in the dropdown. Verify WASM core maintains individual indexing states for all three', async () => {
    const fileInput = document.getElementById('file-input');
    
    const f1 = new File(["red text"], "red.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [f1], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    const docId1 = document.getElementById('doc-selector').value;
    
    const f2 = new File(["blue text"], "blue.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [f2], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    const docId2 = document.getElementById('doc-selector').value;
    
    const f3 = new File(["green text"], "green.txt", { type: "text/plain" });
    Object.defineProperty(fileInput, 'files', { value: [f3], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    const docId3 = document.getElementById('doc-selector').value;
    
    const selector = document.getElementById('doc-selector');
    
    // Switch to f1 and query
    selector.value = docId1;
    selector.dispatchEvent(new Event('change'));
    document.getElementById('query-input').value = "red";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("red");
    
    // Switch to f2 and query
    selector.value = docId2;
    selector.dispatchEvent(new Event('change'));
    document.getElementById('query-input').value = "blue";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("blue");
    
    // Switch to f3 and query
    selector.value = docId3;
    selector.dispatchEvent(new Event('change'));
    document.getElementById('query-input').value = "green";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("green");
  });
});
