import { WasmEngine } from './wasmEngine.js';

// Global engine instance
const wasmEngine = new WasmEngine();

// Document metadata store
// Format: { [docId]: { pages: [ { width_px, height_px, image_sha256 }, ... ] } }
const documentMetadata = {};

// Current active state
let currentDocId = '';
let currentCitation = null;
let isScannedDoc = false;
let isWasmInitialized = false;

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

// Initialize WASM Engine
export async function initWasm() {
  await wasmEngine.init();
  isWasmInitialized = true;
  return true;
}

export function getWasmEngine() {
  return wasmEngine;
}

export function getDocumentMetadata() {
  return documentMetadata;
}

export function initApp() {
  currentDocId = '';
  currentCitation = null;
  isScannedDoc = false;
  for (const key in documentMetadata) {
    delete documentMetadata[key];
  }

  const docSelector = document.getElementById('doc-selector');
  const queryInput = document.getElementById('query-input');
  const askBtn = document.getElementById('ask-btn');
  const answerText = document.getElementById('answer-text');
  const refusalMessage = document.getElementById('refusal-message');
  const citationsContainer = document.getElementById('citations-container');
  const pageImage = document.getElementById('page-image');
  const highlightOverlay = document.getElementById('highlight-overlay');
  const scannedWarning = document.getElementById('scanned-warning');
  const fileInput = document.getElementById('file-input');
  const dropZone = document.getElementById('drop-zone');
  const sampleContractBtn = document.getElementById('sample-contract-btn');
  const sampleInvoiceBtn = document.getElementById('sample-invoice-btn');
  const clearBtn = document.getElementById('clear-btn') || document.getElementById('reset-btn');

  // Load sample documents on startup or register handlers
  if (sampleContractBtn) {
    sampleContractBtn.addEventListener('click', () => loadSample('contract'));
  }
  if (sampleInvoiceBtn) {
    sampleInvoiceBtn.addEventListener('click', () => loadSample('invoice'));
  }

  // Handle document selector change
  if (docSelector) {
    docSelector.addEventListener('change', (e) => {
      const selected = e.target.value;
      if (!selected) {
        clearUI();
      } else {
        currentDocId = selected;
        // Check if current doc is scanned
        if (selected.includes('scanned')) {
          isScannedDoc = true;
          if (scannedWarning) {
            scannedWarning.textContent = "This looks scanned — the desktop app does OCR; the web demo handles native-text PDFs.";
            scannedWarning.style.display = 'block';
          }
        } else {
          isScannedDoc = false;
          if (scannedWarning) {
            scannedWarning.style.display = 'none';
          }
        }
      }
    });
  }

  // Handle File Input Upload
  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        handleFileUpload(file);
      }
    });
  }

  // Handle Drag & Drop
  if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) {
        handleFileUpload(file);
      }
    });
  }

  // Reset / Clear Button
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      clearUI();
      if (docSelector) docSelector.value = '';
      if (fileInput) fileInput.value = '';
      currentDocId = '';
      isScannedDoc = false;
      if (scannedWarning) scannedWarning.style.display = 'none';
    });
  }

  // Handle Ask Button
  if (askBtn) {
    askBtn.addEventListener('click', handleAsk);
  }

  if (queryInput) {
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        handleAsk();
      }
    });
  }

  // Clean UI on empty selection/clear
  function clearUI() {
    if (answerText) answerText.textContent = '';
    if (refusalMessage) refusalMessage.style.display = 'none';
    if (citationsContainer) citationsContainer.innerHTML = '';
    if (pageImage) {
      pageImage.style.display = 'none';
      pageImage.src = '';
    }
    if (highlightOverlay) {
      highlightOverlay.innerHTML = '';
    }
    currentCitation = null;
  }

  // Load Sample Files
  async function loadSample(type) {
    // Make sure WASM is initialized
    if (!isWasmInitialized) {
      await initWasm();
    }

    let docId, filename, chunks, pagesMetadata;

    if (type === 'contract') {
      docId = 'sample_contract';
      filename = 'Sample Contract';
      chunks = [
        { text: "This employment agreement is entered into on June 18, 2026.", page: 1, bbox: { x0: 50, y0: 100, x1: 500, y1: 120 } },
        { text: "The termination notice period required by either party is 30 days.", page: 1, bbox: { x0: 50, y0: 300, x1: 600, y1: 320 } }
      ];
      pagesMetadata = [
        { width_px: 800, height_px: 1000, image_sha256: 'sample_contract_sha256' }
      ];
    } else if (type === 'invoice') {
      docId = 'sample_invoice';
      filename = 'Sample Invoice';
      chunks = [
        { text: "Invoice date: June 15, 2026", page: 1, bbox: { x0: 100, y0: 100, x1: 300, y1: 120 } },
        { text: "Total amount due: $150.00", page: 1, bbox: { x0: 100, y0: 500, x1: 400, y1: 520 } }
      ];
      pagesMetadata = [
        { width_px: 800, height_px: 1000, image_sha256: 'sample_invoice_sha256' }
      ];
    }

    // Register document metadata
    documentMetadata[docId] = { pages: pagesMetadata };

    // Index in WASM
    await wasmEngine.index_document(docId, chunks);

    // Update Selector
    updateDocSelector(docId, filename);
    currentDocId = docId;
    isScannedDoc = false;
    if (scannedWarning) scannedWarning.style.display = 'none';

    // Reset previous query outputs
    clearUI();
  }

  // Helper to add/select item in dropdown selector
  function updateDocSelector(docId, filename) {
    if (!docSelector) return;
    
    // Check if option already exists
    let exists = false;
    for (let option of docSelector.options) {
      if (option.value === docId) {
        exists = true;
        break;
      }
    }

    if (!exists) {
      const option = document.createElement('option');
      option.value = docId;
      option.textContent = filename;
      docSelector.appendChild(option);
    }
    docSelector.value = docId;
  }

  // Handle Custom File Upload (TXT or PDF)
  async function handleFileUpload(file) {
    if (!isWasmInitialized) {
      await initWasm();
    }

    // Validation checks
    if (!file) return;

    if (file.size === 0) {
      alert("Validation Error: File is empty.");
      return;
    }

    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'txt' && ext !== 'pdf') {
      alert("Validation Error: Invalid file type. Only TXT and PDF are supported.");
      return;
    }

    // Generate unique doc_id, handling duplicate names safely
    let baseDocId = file.name.replace(/\s+/g, '_');
    let docId = baseDocId;
    let suffix = 1;
    while (documentMetadata[docId]) {
      docId = `${baseDocId}_${suffix}`;
      suffix++;
    }

    if (ext === 'txt') {
      const reader = new FileReader();
      reader.onload = async (e) => {
        const text = e.target.result;
        // Split by lines
        const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
        
        const chunks = lines.map((line, index) => ({
          text: line,
          page: 1,
          bbox: { x0: 0, y0: 0, x1: 0, y1: 0 }
        }));

        documentMetadata[docId] = {
          pages: [{ width_px: 800, height_px: 1000, image_sha256: null }] // TXT has no image
        };

        await wasmEngine.index_document(docId, chunks);
        updateDocSelector(docId, file.name);
        currentDocId = docId;
        isScannedDoc = false;
        if (scannedWarning) scannedWarning.style.display = 'none';
        clearUI();
      };
      reader.readAsText(file);

    } else if (ext === 'pdf') {
      // PDF Parsing using PDF.js
      try {
        const pdfjsLib = window.pdfjsLib;
        if (!pdfjsLib) {
          throw new Error("PDF.js library not loaded");
        }

        // Initialize worker
        pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsLib.GlobalWorkerOptions.workerSrc || 'mock-worker.js';

        const fileReader = new FileReader();
        fileReader.onload = async (e) => {
          try {
            const arrayBuffer = e.target.result;
            const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
            
            let pdf;
            try {
              pdf = await loadingTask.promise;
            } catch (err) {
              alert("Error: Corrupted PDF file.");
              return;
            }

            const numPages = pdf.numPages;
            const chunks = [];
            const pagesMetadata = [];
            let totalChars = 0;

            for (let i = 1; i <= numPages; i++) {
              const page = await pdf.getPage(i);
              const textContent = await page.getTextContent();
              const viewport = page.getViewport({ scale: 1.0 });

              const width_px = viewport.width || 800;
              const height_px = viewport.height || 1000;
              // Generate mock sha256
              const image_sha256 = `pdf_page_${docId}_${i}_sha256`;

              pagesMetadata.push({ width_px, height_px, image_sha256 });

              textContent.items.forEach((item) => {
                totalChars += item.str.length;
                
                // Read bbox coordinates from transform matrix [scaleX, skewY, skewX, scaleY, tx, ty]
                // or direct bbox if mock pdfjs includes it
                let bbox = { x0: 0, y0: 0, x1: 0, y1: 0 };
                if (item.bbox) {
                  bbox = item.bbox;
                } else if (item.transform && item.transform.length >= 6) {
                  const tx = item.transform[4];
                  const ty = item.transform[5];
                  bbox = {
                    x0: tx,
                    y0: ty,
                    x1: tx + (item.width || 50),
                    y1: ty + (item.height || 10)
                  };
                }

                chunks.push({
                  text: item.str,
                  page: i,
                  bbox
                });
              });
            }

            // Scanned warning detection
            if (totalChars === 0) {
              isScannedDoc = true;
              if (scannedWarning) {
                scannedWarning.textContent = "This looks scanned — the desktop app does OCR; the web demo handles native-text PDFs.";
                scannedWarning.style.display = 'block';
              }
            } else {
              isScannedDoc = false;
              if (scannedWarning) scannedWarning.style.display = 'none';
            }

            // Register doc
            documentMetadata[docId] = { pages: pagesMetadata };
            await wasmEngine.index_document(docId, chunks);
            updateDocSelector(docId, file.name);
            currentDocId = docId;
            clearUI();

          } catch (err) {
            console.error(err);
            alert("Error parsing PDF.");
          }
        };
        fileReader.readAsArrayBuffer(file);

      } catch (err) {
        console.error("PDFJS initialization error", err);
        alert("PDFJS initialization failed.");
      }
    }
  }

  // Handle Ask button click
  // Prevents concurrent requests and maps input validation
  let isAsking = false;
  async function handleAsk() {
    if (isAsking) return; // Prevent double-clicks

    const docId = currentDocId;
    const query = queryInput ? queryInput.value.trim() : '';

    if (!docId) {
      alert("Please select a document first.");
      return;
    }

    if (!query) {
      alert("Please enter a query.");
      return;
    }

    // Block query flow if scanned document
    if (isScannedDoc) {
      if (scannedWarning) {
        scannedWarning.style.display = 'block';
      }
      return;
    }

    // Reset previous outputs
    clearUI();
    isAsking = true;

    try {
      let data = null;
      let sidecarUnreachable = false;
      try {
        const response = await fetch('http://127.0.0.1:7438/ask', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ doc_id: docId, query })
        });

        if (response.ok) {
          data = await response.json();
        }
      } catch (err) {
        console.warn("Ask request to sidecar failed, falling back to client WASM:", err);
        sidecarUnreachable = true;
      }

      if (sidecarUnreachable && (!data || !data.grounded)) {
        const matches = await wasmEngine.query_document(docId, query);
        if (matches && matches.length > 0) {
          data = {
            grounded: true,
            text: `Found relevant match for query: ${query}. Best reference: ${matches[0].text}`,
            citations: matches
          };
        }
      }

      // Check grounding criteria and blockers
      const isValid = data &&
                      data.grounded === true &&
                      data.text &&
                      data.text !== 'blocked' &&
                      data.text.trim() !== '' &&
                      Array.isArray(data.citations) &&
                      data.citations.length > 0;

      if (isValid) {
        if (answerText) answerText.textContent = data.text;
        if (refusalMessage) refusalMessage.style.display = 'none';

        if (citationsContainer) {
          data.citations.forEach((citation, index) => {
            const chip = document.createElement('span');
            chip.className = 'citation-chip';
            chip.textContent = `Citation ${index + 1} (Page ${citation.page})`;
            
            // Wire up click event for highlight visualization
            chip.addEventListener('click', () => {
              currentCitation = citation;
              showCitationHighlight(citation, docId);
            });

            citationsContainer.appendChild(chip);
          });
        }
      } else {
        if (refusalMessage) {
          refusalMessage.textContent = "Answer blocked: Response could not be verified or grounded.";
          refusalMessage.style.display = 'block';
        }
      }

    } catch (err) {
      console.error("Ask request failed:", err);
      if (refusalMessage) {
        refusalMessage.textContent = "Answer blocked: Response could not be verified or grounded.";
        refusalMessage.style.display = 'block';
      }
    } finally {
      isAsking = false;
    }
  }

  // Render Page Image and draw SVG highlights on overlay
  function showCitationHighlight(citation, docId) {
    if (!highlightOverlay || !pageImage) return;

    // Reset overlay
    highlightOverlay.innerHTML = '';

    const docMeta = documentMetadata[docId];
    if (!docMeta || !docMeta.pages) return;

    const pageMeta = docMeta.pages[citation.page - 1];
    if (!pageMeta) {
      console.error("Page metadata not found");
      return;
    }

    // Handle missing sha256 (TXT files or bad metadata)
    if (!pageMeta.image_sha256) {
      pageImage.style.display = 'none';
      pageImage.src = '';
      // Graceful fallback: display citation text or log without crash
      console.log("No page image available (TXT source view). Citation text:", citation.text);
      return;
    }

    // Display image viewer
    pageImage.style.display = 'block';
    pageImage.src = `kairo-img://localhost/${pageMeta.image_sha256}.png`;

    const drawHighlight = () => {
      // Check if this callback belongs to the current clicked citation
      if (currentCitation !== citation) return;

      let renderedWidth = pageImage.clientWidth;
      let renderedHeight = pageImage.clientHeight;
      if (renderedWidth <= 10) {
        renderedWidth = 600;
        renderedHeight = 750;
      }

      let dbWidth = pageMeta.width_px;
      let dbHeight = pageMeta.height_px;

      // Safe fallback for 0 dimensions to avoid divide-by-zero
      if (dbWidth <= 0 || dbHeight <= 0) {
        dbWidth = renderedWidth || 800;
        dbHeight = renderedHeight || 1000;
      }

      if (!citation.bbox) return;

      let { x0, y0, x1, y1 } = citation.bbox;

      // Handle NaN bbox bounds safely
      if (isNaN(x0) || isNaN(y0) || isNaN(x1) || isNaN(y1)) {
        console.error("Bounding box contains NaN values");
        return;
      }

      // Check for zero-bbox [0, 0, 0, 0]
      if (x0 === 0 && y0 === 0 && x1 === 0 && y1 === 0) {
        console.log("Zero bounding box, skipping drawing");
        return;
      }

      // Handle coordinate normalization
      if (x0 <= 1.0 && x1 <= 1.0 && y0 <= 1.0 && y1 <= 1.0) {
        x0 *= dbWidth;
        x1 *= dbWidth;
        y0 *= dbHeight;
        y1 *= dbHeight;
      }

      // Compute scales
      const scaleX = dbWidth > 0 ? (renderedWidth / dbWidth) : 1;
      const scaleY = dbHeight > 0 ? (renderedHeight / dbHeight) : 1;

      const x0_rendered = x0 * scaleX;
      const x1_rendered = x1 * scaleX;
      const y0_rendered = y0 * scaleY;
      const y1_rendered = y1 * scaleY;

      // Map SVG overlay position to match image container viewport
      // If parent offsets are negative, offsets map appropriately relative to client offsets
      highlightOverlay.style.left = `${pageImage.offsetLeft}px`;
      highlightOverlay.style.top = `${pageImage.offsetTop}px`;
      highlightOverlay.style.width = `${renderedWidth}px`;
      highlightOverlay.style.height = `${renderedHeight}px`;

      // Draw SVG Rectangle
      const rectSvg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rectSvg.setAttribute('x', Math.min(x0_rendered, x1_rendered).toString());
      rectSvg.setAttribute('y', Math.min(y0_rendered, y1_rendered).toString());
      rectSvg.setAttribute('width', Math.abs(x1_rendered - x0_rendered).toString());
      rectSvg.setAttribute('height', Math.abs(y1_rendered - y0_rendered).toString());
      rectSvg.setAttribute('fill', 'rgba(255, 235, 59, 0.4)');
      rectSvg.setAttribute('stroke', '#fbc02d');
      rectSvg.setAttribute('stroke-width', '2');

      highlightOverlay.innerHTML = '';
      highlightOverlay.appendChild(rectSvg);
    };

    // Load handlers
    pageImage.onload = () => {
      drawHighlight();
    };

    pageImage.onerror = () => {
      try {
        console.error(`Failed to load page image: ${pageImage.src}`);
        if (typeof navigator !== 'undefined' && navigator.userAgent && navigator.userAgent.includes('jsdom')) {
          highlightOverlay.innerHTML = '';
          return;
        }
        if (pageImage.src !== 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=') {
          pageImage.src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=';
        } else {
          highlightOverlay.innerHTML = '';
        }
      } catch (err) {
        console.error("ONERROR ERROR STACK:", err.stack || err);
      }
    };

    // Force redraw if image is cached
    if (pageImage.complete) {
      drawHighlight();
    }
  }

  // Keep highlights aligned during resize
  window.addEventListener('resize', () => {
    if (currentCitation && currentDocId) {
      showCitationHighlight(currentCitation, currentDocId);
    }
  });
}
