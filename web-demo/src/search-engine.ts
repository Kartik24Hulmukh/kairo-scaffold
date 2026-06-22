export interface Citation {
  chunk_id: string;
  page: number;
  bbox: number[];
  text: string;
}

export interface QueryResult {
  grounded: boolean;
  answer: string;
  citations: Citation[];
}

let wasmModule: any = null;
let isWasmLoaded = false;

// Mock database for JS fallback
const mockDocuments: Record<string, string> = {};

export async function initSearchEngine() {
  try {
    // Attempt dynamic import of compiled WASM core pkg
    // @ts-ignore
    const wasm = await import('./pkg/wasm_search_core.js');
    await wasm.default();
    wasm.init_engine();
    wasmModule = wasm;
    isWasmLoaded = true;
    console.log("Rust WASM search core initialized successfully.");
  } catch (e) {
    console.warn("WASM core failed to load, falling back to local JS DB:", e);
    isWasmLoaded = false;
  }
}

export function indexDocument(docId: string, content: string): boolean {
  if (isWasmLoaded && wasmModule) {
    return wasmModule.index_document(docId, content);
  }
  mockDocuments[docId] = content;
  return true;
}

export function queryEngine(docId: string, query: string): QueryResult {
  if (isWasmLoaded && wasmModule) {
    const rawResult = wasmModule.query_engine(docId, query);
    return JSON.parse(rawResult);
  }

  // JS Fallback matching
  const docContent = mockDocuments[docId] || "";
  if (!docContent) {
    return {
      grounded: false,
      answer: "No document indexed.",
      citations: []
    };
  }

  // Simple substring/keyword match for stub fallback
  const normalizedQuery = query.toLowerCase();
  const lines = docContent.split('\n');
  const matchedLines = lines.filter(line => line.toLowerCase().includes(normalizedQuery));

  if (matchedLines.length > 0) {
    return {
      grounded: true,
      answer: `Found match for: ${query}. Details: ${matchedLines[0]}`,
      citations: [
        {
          chunk_id: "chunk_0",
          page: 1,
          bbox: [50, 100, 250, 120],
          text: matchedLines[0]
        }
      ]
    };
  }

  return {
    grounded: false,
    answer: "I am sorry, but the query could not be answered from the document source.",
    citations: []
  };
}
