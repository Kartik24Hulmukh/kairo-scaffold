import { vi } from 'vitest';

let askMockOverride = null;

export function setAskMockOverride(override) {
  askMockOverride = override;
}

export function clearAskMockOverride() {
  askMockOverride = null;
}

export function setupDOM() {
  document.body.innerHTML = `
    <div id="drop-zone" style="width: 200px; height: 200px; border: 1px solid black;">Drag files here</div>
    <input type="file" id="file-input" />
    <button id="sample-contract-btn">Sample Contract</button>
    <button id="sample-invoice-btn">Sample Invoice</button>
    <button id="clear-btn">Reset</button>

    <select id="doc-selector">
      <option value="">Select Document</option>
    </select>

    <div id="scanned-warning" style="display: none;"></div>

    <input id="query-input" />
    <button id="ask-btn">Ask</button>

    <div id="answer-text"></div>
    <div id="refusal-message" style="display: none;">Answer blocked: Response could not be verified or grounded.</div>
    <div id="citations-container"></div>

    <div id="viewer-container" style="position: relative; width: 800px; height: 1000px;">
      <img id="page-image" style="display: none; width: 100%; height: 100%;" />
      <svg id="highlight-overlay" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%;"></svg>
    </div>
  `;

  // Set initial mock offsets/dimensions for JSDOM
  const pageImage = document.getElementById('page-image');
  const highlightOverlay = document.getElementById('highlight-overlay');
  
  if (pageImage) {
    Object.defineProperties(pageImage, {
      clientWidth: { get: () => 800, configurable: true },
      clientHeight: { get: () => 1000, configurable: true },
      offsetLeft: { get: () => 10, configurable: true },
      offsetTop: { get: () => 20, configurable: true },
      complete: { get: () => true, configurable: true }
    });
  }
}

export function setupPdfJsMock() {
  window.pdfjsLib = {
    GlobalWorkerOptions: {
      workerSrc: ''
    },
    getDocument: ({ data }) => {
      // Decode data (which is a Uint8Array or ArrayBuffer)
      const decoder = new TextDecoder();
      const str = decoder.decode(data);
      
      try {
        const parsed = JSON.parse(str);
        if (parsed.corrupt) {
          throw new Error("Corrupted PDF signature");
        }
        
        return {
          promise: Promise.resolve({
            numPages: parsed.pages.length,
            getPage: async (pageNum) => {
              const pageData = parsed.pages[pageNum - 1];
              if (!pageData) throw new Error(`Page ${pageNum} not found`);
              return {
                getTextContent: async () => {
                  return {
                    items: pageData.items.map(item => ({
                      str: item.str,
                      bbox: item.bbox || { x0: 0, y0: 0, x1: 0, y1: 0 },
                      transform: item.transform || [1, 0, 0, 1, 0, 0]
                    }))
                  };
                },
                getViewport: ({ scale }) => {
                  return {
                    width: pageData.width || 800,
                    height: pageData.height || 1000
                  };
                }
              };
            }
          })
        };
      } catch (e) {
        // Return a promise that rejects to simulate PDF.js parsing error
        return {
          promise: Promise.reject(new Error("PDF.js loading failed: Corrupted file structure"))
        };
      }
    }
  };
}

import fs from 'fs';
import path from 'path';

export function setupFetchMock(wasmEngine) {
  const originalFetch = window.fetch;
  window.fetch = vi.fn(async (url, init) => {
    const urlStr = typeof url === 'string' ? url : (url.url || url.href || String(url));
    
    if (urlStr.endsWith('.wasm')) {
      try {
        const wasmPath = path.resolve(process.cwd(), 'src/pkg/wasm_search_core_bg.wasm');
        const buffer = fs.readFileSync(wasmPath);
        return new Response(buffer, {
          status: 200,
          headers: { 'Content-Type': 'application/wasm' }
        });
      } catch (e) {
        console.error("Failed to read wasm file in fetch mock:", e);
        return { ok: false, status: 404 };
      }
    }

    if (urlStr.includes('/ask')) {
      if (askMockOverride) {
        if (typeof askMockOverride === 'function') {
          return askMockOverride(init);
        }
        return {
          ok: askMockOverride.ok !== undefined ? askMockOverride.ok : true,
          status: askMockOverride.status || 200,
          json: async () => askMockOverride.json
        };
      }
      
      const body = JSON.parse(init.body);
      const matches = await wasmEngine.query_document(body.doc_id, body.query);
      
      if (matches.length > 0) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            grounded: true,
            text: `Grounded answer: found matches for '${body.query}'.`,
            citations: matches.map(m => ({
              page: m.page,
              bbox: m.bbox,
              text: m.text
            }))
          })
        };
      } else {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            grounded: false,
            text: "No matches found.",
            citations: []
          })
        };
      }
    }
    
    if (originalFetch) {
      return originalFetch(url, init);
    }
    return { ok: false, status: 404 };
  });
}

