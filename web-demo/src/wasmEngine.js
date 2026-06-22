let wasmModule = null;
let isWasmLoaded = false;

export class WasmEngine {
  constructor() {
    this.documents = new Map();
    this.initialized = false;
  }

  async init() {
    this.initialized = true;
    try {
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
    return true;
  }

  async index_document(doc_id, chunks) {
    if (!this.initialized) {
      throw new Error("WasmEngine not initialized");
    }
    if (doc_id === undefined || doc_id === null || doc_id === "") {
      throw new Error("Invalid doc_id");
    }
    if (!chunks) {
      throw new Error("Invalid chunks");
    }

    // Keep track of document chunks locally
    this.documents.set(doc_id, chunks);

    if (isWasmLoaded && wasmModule) {
      const success = wasmModule.index_document(doc_id, JSON.stringify(chunks));
      return {
        doc_id,
        chunk_count: chunks.length,
        success,
        summary: `Indexed ${chunks.length} chunks for document ${doc_id}`
      };
    }

    const chunk_count = chunks.length;
    return {
      doc_id,
      chunk_count,
      success: true,
      summary: `Indexed ${chunk_count} chunks for document ${doc_id}`
    };
  }

  async query_document(doc_id, query_term) {
    if (!this.initialized) {
      throw new Error("WasmEngine not initialized");
    }
    if (!doc_id) {
      return [];
    }
    if (!query_term || query_term.trim() === "") {
      return [];
    }

    if (isWasmLoaded && wasmModule) {
      try {
        const rawResult = wasmModule.query_engine(doc_id, query_term);
        const result = JSON.parse(rawResult);
        if (result.grounded) {
          return result.citations;
        }
        return [];
      } catch (err) {
        console.error("Rust WASM query_engine error, falling back:", err);
      }
    }

    const chunks = this.documents.get(doc_id);
    if (!chunks) {
      return [];
    }

    // Perform case-insensitive substring search on chunk text
    const lowerQuery = query_term.toLowerCase();
    const matches = chunks.filter(chunk => {
      if (!chunk.text) return false;
      return chunk.text.toLowerCase().includes(lowerQuery);
    });

    return matches;
  }

  clear() {
    this.documents.clear();
    if (isWasmLoaded && wasmModule && typeof wasmModule.clear_engine === 'function') {
      wasmModule.clear_engine();
    }
  }
}

// Expose on window for easy access in scripts/tests
if (typeof window !== "undefined") {
  window.WasmEngine = WasmEngine;
}

