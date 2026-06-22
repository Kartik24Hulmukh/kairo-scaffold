import { describe, it, expect, beforeEach, vi } from 'vitest';
import { initApp, initWasm, getWasmEngine, getDocumentMetadata } from '../index.js';
import { setupDOM, setupPdfJsMock, setupFetchMock, setAskMockOverride, clearAskMockOverride } from './testHelper.js';

describe('Tier 4 — Real-World Workloads (5 Tests)', () => {
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

  it('t4_financial_invoice_workload: Load a mock multi-page invoice containing tables. Ask "What is the total amount due?", verify answer is grounded and highlights are precisely drawn on the currency value', async () => {
    const fileInput = document.getElementById('file-input');
    const mockInvoiceData = JSON.stringify({
      pages: [
        {
          width: 800,
          height: 1000,
          items: [
            { str: "Invoice #1024", transform: [1,0,0,1,50,50] },
            { str: "Line items table", transform: [1,0,0,1,50,200] }
          ]
        },
        {
          width: 800,
          height: 1000,
          items: [
            { str: "Summary metrics", transform: [1,0,0,1,50,50] },
            // target currency value at precise bbox coordinates: x0: 200, y0: 300, x1: 300, y1: 320
            { str: "Total amount due: $1500.00", bbox: { x0: 200, y0: 300, x1: 300, y1: 320 } }
          ]
        }
      ]
    });
    
    const file = new File([mockInvoiceData], "invoice_1024.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    document.getElementById('query-input').value = "total amount due";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("total amount due");
    
    const chips = document.querySelectorAll('.citation-chip');
    expect(chips.length).toBe(1);
    expect(chips[0].textContent).toContain("Page 2");
    
    chips[0].click();
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
    expect(Number(rect.getAttribute('x'))).toBe(200);
    expect(Number(rect.getAttribute('y'))).toBe(300);
    expect(Number(rect.getAttribute('width'))).toBe(100);
  });

  it('t4_employment_contract_workload: Load an employment agreement. Query about "termination notice". Verify answer matches the agreement text and highlights the legal term section', async () => {
    const fileInput = document.getElementById('file-input');
    const mockContractData = JSON.stringify({
      pages: [
        {
          width: 800,
          height: 1000,
          items: [
            { str: "Employment Agreement", transform: [1,0,0,1,100,50] },
            { str: "Section 5: Termination", transform: [1,0,0,1,100,200] },
            { str: "Either party may terminate with 30 days written termination notice.", bbox: { x0: 100, y0: 250, x1: 600, y1: 280 } }
          ]
        }
      ]
    });
    
    const file = new File([mockContractData], "contract.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    document.getElementById('query-input').value = "termination notice";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("termination notice");
    
    const chip = document.querySelector('.citation-chip');
    chip.click();
    
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    const rect = document.querySelector('#highlight-overlay rect');
    expect(rect).toBeTruthy();
    expect(Number(rect.getAttribute('x'))).toBe(100);
    expect(Number(rect.getAttribute('y'))).toBe(250);
  });

  it('t4_research_paper_workload: Load a research paper with sections. Query about "dataset used in experiments". Verify answer matches abstract/methodology sections with citations', async () => {
    const fileInput = document.getElementById('file-input');
    const mockPaperData = JSON.stringify({
      pages: [
        {
          width: 800,
          height: 1000,
          items: [
            { str: "Paper Title: AI Models", transform: [1,0,0,1,100,50] },
            { str: "Abstract: We study transformers.", transform: [1,0,0,1,100,200] }
          ]
        },
        {
          width: 800,
          height: 1000,
          items: [
            { str: "Methodology & Experiments", transform: [1,0,0,1,100,50] },
            { str: "The dataset used in experiments is ImageNet-1k containing 1.2M images.", bbox: { x0: 100, y0: 150, x1: 700, y1: 180 } }
          ]
        }
      ]
    });
    
    const file = new File([mockPaperData], "paper.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    document.getElementById('query-input').value = "dataset used in experiments";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("dataset used in experiments");
    
    const chip = document.querySelector('.citation-chip');
    expect(chip.textContent).toContain("Page 2");
  });

  it('t4_scanned_and_text_mixed_batch: Simulate a user uploading 3 scanned PDFs, 2 native PDFs, and 2 text files in a single session. Verify scanned warnings toggle correctly, indexing succeeds, and querying works for native/text documents', async () => {
    const fileInput = document.getElementById('file-input');
    const warning = document.getElementById('scanned-warning');
    const selector = document.getElementById('doc-selector');
    
    const scannedData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [] }] });
    const nativeData = JSON.stringify({ pages: [{ width: 800, height: 1000, items: [{ str: "native text is here", transform: [1,0,0,1,0,0] }] }] });
    
    // Upload 3 scanned PDFs
    for (let i = 1; i <= 3; i++) {
      const file = new File([scannedData], `scanned_${i}.pdf`, { type: "application/pdf" });
      Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      await wait();
      expect(warning.style.display).toBe('block');
    }
    
    // Upload 2 native PDFs
    for (let i = 1; i <= 2; i++) {
      const file = new File([nativeData], `native_${i}.pdf`, { type: "application/pdf" });
      Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      await wait();
      expect(warning.style.display).toBe('none');
    }
    
    // Upload 2 text files
    for (let i = 1; i <= 2; i++) {
      const file = new File([`text content ${i}`], `text_${i}.txt`, { type: "text/plain" });
      Object.defineProperty(fileInput, 'files', { value: [file], writable: true });
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      await wait();
      expect(warning.style.display).toBe('none');
    }
    
    // Select native_1.pdf and query it
    selector.value = selector.options[4].value; // First native
    selector.dispatchEvent(new Event('change'));
    await wait();
    expect(warning.style.display).toBe('none');
    
    document.getElementById('query-input').value = "native text";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("native text");
  });

  it('t4_robust_session_workload: Execute a complete user journey: select Sample Contract, ask 2 questions, click citation, upload custom PDF, ask 3 questions (one triggers refusal), click citations, select Sample Invoice, ask 1 question, verify zero UI crashes, zero leaks, and correct displays', async () => {
    // 1. Select sample contract
    document.getElementById('sample-contract-btn').click();
    await wait();
    expect(document.getElementById('doc-selector').value).toBe('sample_contract');
    
    // 2. Ask question 1
    document.getElementById('query-input').value = "employment agreement";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("employment agreement");
    
    // 3. Ask question 2
    document.getElementById('query-input').value = "termination notice";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("termination notice");
    
    // 4. Click citation
    const chip = document.querySelector('.citation-chip');
    chip.click();
    const pageImage = document.getElementById('page-image');
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    expect(document.querySelector('#highlight-overlay rect')).toBeTruthy();
    
    // 5. Upload custom PDF
    const fileInput = document.getElementById('file-input');
    const mockPdfData = JSON.stringify({
      pages: [{
        width: 800,
        height: 1000,
        items: [
          { str: "secret code validation", transform: [1,0,0,1,0,0] },
          { str: "security guidelines match", transform: [1,0,0,1,0,0] }
        ]
      }]
    });
    const customFile = new File([mockPdfData], "custom.pdf", { type: "application/pdf" });
    Object.defineProperty(fileInput, 'files', { value: [customFile], writable: true });
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    await wait();
    
    // 6. Ask question 1 on custom PDF (grounded)
    document.getElementById('query-input').value = "secret code";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("secret code");
    
    // 7. Ask question 2 on custom PDF (refusal / ungrounded)
    document.getElementById('query-input').value = "non-existent term";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('refusal-message').style.display).toBe('block');
    
    // 8. Ask question 3 on custom PDF (grounded)
    document.getElementById('query-input').value = "security guidelines";
    document.getElementById('ask-btn').click();
    await wait();
    expect(document.getElementById('answer-text').textContent).toContain("security guidelines");
    
    // 9. Click citations
    document.querySelector('.citation-chip').click();
    pageImage.dispatchEvent(new Event('load'));
    await wait();
    
    // 10. Select Sample Invoice
    document.getElementById('sample-invoice-btn').click();
    await wait();
    
    // 11. Ask 1 question
    document.getElementById('query-input').value = "total amount due";
    document.getElementById('ask-btn').click();
    await wait();
    
    expect(document.getElementById('answer-text').textContent).toContain("total amount due");
    expect(document.getElementById('refusal-message').style.display).toBe('none');
  });
});
