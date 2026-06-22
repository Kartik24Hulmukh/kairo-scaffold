import { useEffect } from 'react';
// @ts-ignore
import { initApp, initWasm } from './index.js';
import { Upload, HelpCircle, FileText, CheckCircle2, AlertTriangle, AlertCircle, RefreshCw } from 'lucide-react';

export default function App() {
  useEffect(() => {
    const startApp = async () => {
      try {
        await initWasm();
        initApp();
      } catch (err) {
        console.error("Failed to initialize app/WASM:", err);
      }
    };
    startApp();
  }, []);

  return (
    <div className="min-h-screen bg-[#060713] text-slate-100 font-sans p-6 md:p-12 flex flex-col items-center selection:bg-purple-500/30 selection:text-purple-200">
      {/* Background Glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-600/10 rounded-full blur-[100px] pointer-events-none" />

      <header className="mb-12 text-center max-w-xl z-10">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-indigo-400 to-purple-400 bg-clip-text text-transparent flex items-center justify-center gap-3">
          Kairo <span className="text-sm font-semibold tracking-widest uppercase bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2.5 py-0.5 rounded-full">Web WASM</span>
        </h1>
        <p className="text-slate-400 mt-3 text-lg leading-relaxed">
          Stateless client-side document search, parsing, and citation highlighting powered by Rust compiled to WebAssembly.
        </p>
      </header>

      <main className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-12 gap-8 z-10 items-start">
        {/* Left Control Panel */}
        <section className="lg:col-span-4 flex flex-col gap-6 w-full">
          {/* Upload Card */}
          <div className="glass-card">
            <h2 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center gap-2">
              <Upload className="w-5 h-5" /> Ingest Document
            </h2>

            {/* Drag & Drop Zone */}
            <div 
              id="drop-zone" 
              className="border border-dashed border-slate-700/60 hover:border-cyan-500/60 hover:bg-cyan-500/[0.02] rounded-xl p-8 text-center cursor-pointer transition-all duration-300 group flex flex-col items-center justify-center gap-3"
              onClick={() => document.getElementById('file-input')?.click()}
            >
              <div className="w-12 h-12 rounded-full bg-slate-800/80 border border-slate-700/50 flex items-center justify-center group-hover:scale-110 group-hover:border-cyan-500/40 transition-all duration-300">
                <Upload className="w-5 h-5 text-slate-400 group-hover:text-cyan-400 transition-colors" />
              </div>
              <div>
                <p className="text-sm font-medium text-slate-300">Drag & drop document</p>
                <p className="text-xs text-slate-500 mt-1">PDF or TXT up to 10MB</p>
              </div>
              <input type="file" id="file-input" className="hidden" accept=".pdf,.txt" />
            </div>

            {/* One-Click Samples */}
            <div className="mt-6">
              <span className="text-xs font-semibold tracking-wider text-slate-500 uppercase block mb-3">One-Click Samples</span>
              <div className="flex flex-col gap-2">
                <button 
                  id="sample-contract-btn"
                  className="flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/40 rounded-lg text-sm text-slate-300 font-medium transition-all duration-200"
                >
                  <span className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-indigo-400" /> sample_contract.txt
                  </span>
                  <span className="text-xs text-slate-500">Contract</span>
                </button>
                <button 
                  id="sample-invoice-btn"
                  className="flex items-center justify-between p-3 bg-slate-800/40 hover:bg-slate-800/80 border border-slate-700/40 rounded-lg text-sm text-slate-300 font-medium transition-all duration-200"
                >
                  <span className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-purple-400" /> sample_invoice.txt
                  </span>
                  <span className="text-xs text-slate-500">Invoice</span>
                </button>
              </div>
            </div>
          </div>

          {/* Document Selector & Actions */}
          <div className="glass-card">
            <h2 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center gap-2">
              <FileText className="w-5 h-5" /> Document Scope
            </h2>
            <div className="flex flex-col gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase mb-2">Active Document</label>
                <select 
                  id="doc-selector" 
                  className="w-full bg-[#0a0c16]/80 border border-slate-700/60 rounded-lg p-3 text-sm text-slate-200 outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/20 transition-all"
                >
                  <option value="">Select Document</option>
                </select>
              </div>
              <button 
                id="clear-btn" 
                className="w-full py-2.5 px-4 bg-slate-800/50 hover:bg-slate-800 border border-slate-700/40 rounded-lg text-sm font-semibold text-slate-400 hover:text-slate-200 transition-all duration-200 flex items-center justify-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> Reset Workspace
              </button>
            </div>
          </div>
        </section>

        {/* Right Content Panels */}
        <section className="lg:col-span-8 flex flex-col gap-6 w-full">
          {/* Query & Answer Panel */}
          <div className="glass-card">
            <h2 className="text-lg font-semibold text-cyan-400 mb-4 flex items-center gap-2">
              <HelpCircle className="w-5 h-5" /> Ask Document
            </h2>

            {/* Warnings */}
            <div 
              id="scanned-warning" 
              className="hidden p-4 mb-5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-sm flex items-start gap-3"
            >
              <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
              <span />
            </div>

            {/* Input Bar */}
            <div className="flex gap-3">
              <input 
                id="query-input" 
                type="text" 
                placeholder="Ask a question about the active document..."
                className="flex-1 bg-[#0a0c16]/80 border border-slate-700/60 rounded-lg p-3.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-cyan-500/60 focus:ring-1 focus:ring-cyan-500/20 transition-all"
              />
              <button 
                id="ask-btn"
                className="px-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold rounded-lg text-sm shadow-lg shadow-indigo-600/10 hover:shadow-indigo-600/20 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 flex items-center gap-2"
              >
                Query
              </button>
            </div>

            {/* Response Section */}
            <div className="mt-8 border-t border-slate-800/60 pt-6">
              <div id="answer-container" className="flex flex-col gap-4">
                {/* Answer Display */}
                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-500/80" /> Response
                  </h3>
                  <div 
                    id="answer-text" 
                    className="text-slate-200 leading-relaxed text-sm bg-slate-900/30 border border-slate-800/40 p-4 rounded-xl min-h-[60px]"
                  />
                </div>

                {/* Refusal Warning */}
                <div 
                  id="refusal-message" 
                  className="hidden p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm flex items-start gap-3"
                >
                  <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                  <span />
                </div>

                {/* Citations Container */}
                <div>
                  <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Citations</h3>
                  <div 
                    id="citations-container" 
                    className="flex flex-wrap gap-2 min-h-[30px]"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Document Viewer / Canvas Panel */}
          <div className="glass-card flex flex-col">
            <h2 className="text-lg font-semibold text-cyan-400 mb-4">Document Viewer</h2>
            <div className="bg-[#030409]/60 border border-slate-800/60 rounded-xl overflow-hidden flex items-center justify-center p-4 min-h-[350px]">
              <div id="viewer-container" className="relative max-w-full shadow-2xl">
                <img 
                  id="page-image" 
                  className="hidden max-w-full rounded border border-slate-800/50" 
                  alt="Document Page" 
                />
                <svg 
                  id="highlight-overlay" 
                  className="absolute left-0 top-0 w-full h-full pointer-events-none" 
                />
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
