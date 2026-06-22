import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock Tauri invoke
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn().mockResolvedValue([
    { doc_id: 'doc_123', source_path: 'path/to/doc_123.pdf' }
  ])
}));

describe('Kairo Overlay UI Grounding Fuzz Test', () => {
  beforeEach(async () => {
    // Reset DOM
    document.body.innerHTML = `
      <div class="overlay-container glass">
        <select id="doc-selector">
          <option value="">-- Select Document --</option>
          <option value="doc_123">doc_123.pdf</option>
        </select>
        <input type="text" id="query-input" value="test query" />
        <button id="ask-btn">Ask</button>
        <div id="answer-text" class="answer-display">Original Answer</div>
        <div id="refusal-message" class="refusal-display" style="display: none;">Blocked</div>
        <div id="citations-container" class="citations-chips"></div>
        <img id="page-image" style="display: none;" />
        <svg id="highlight-overlay"></svg>
      </div>
    `;

    // Mock fetch
    global.fetch = vi.fn();

    // Reset module registry and import index.js to attach listeners to DOM
    vi.resetModules();
    await import('./src/index.js');
    document.dispatchEvent(new Event('DOMContentLoaded'));
  });

  const runAsk = async (mockResponsePayload) => {
    global.fetch.mockResolvedValue({
      ok: true,
      json: async () => mockResponsePayload
    });

    const docSelector = document.getElementById('doc-selector');
    docSelector.value = 'doc_123';
    const queryInput = document.getElementById('query-input');
    queryInput.value = 'test query';

    const askBtn = document.getElementById('ask-btn');
    askBtn.click();

    // Wait for async request and state updates
    await new Promise(resolve => setTimeout(resolve, 20));
  };

  it('should render the answer and citations when grounded is true and citations exist', async () => {
    const payload = {
      id: 'ans_123',
      query: 'test query',
      text: 'This is a grounded answer.',
      citations: [
        {
          chunk_id: 'chunk_1',
          char_span: [0, 10],
          page: 1,
          bbox: { x0: 0, y0: 0, x1: 0.5, y1: 0.5 }
        }
      ],
      grounded: true
    };

    await runAsk(payload);

    const answerText = document.getElementById('answer-text');
    const refusalMessage = document.getElementById('refusal-message');
    const citationsContainer = document.getElementById('citations-container');

    expect(answerText.textContent).toBe('This is a grounded answer.');
    expect(refusalMessage.style.display).toBe('none');
    expect(citationsContainer.children.length).toBe(1);
    expect(citationsContainer.children[0].textContent).toContain('Citation 1 (Page 1)');
  });

  it('should block and refuse to display when grounded is false, citations are empty, or text is blocked', async () => {
    const fuzzedPayloads = [
      { grounded: false, text: 'Some answer', citations: [] },
      { grounded: false, text: 'Some answer', citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] },
      { grounded: true, text: 'blocked', citations: [] },
      { grounded: true, text: 'Some answer', citations: [] }, // lacks citation/anchor!
      { grounded: true, text: '', citations: [] },
      { grounded: true, text: 'blocked', citations: [{ page: 1, bbox: { x0: 0, y0: 0, x1: 1, y1: 1 } }] },
      { grounded: null, text: 'text', citations: [] },
      { grounded: undefined, text: 'text', citations: [] },
      {}, // empty object
    ];

    for (const payload of fuzzedPayloads) {
      const answerTextEl = document.getElementById('answer-text');
      const refusalMessageEl = document.getElementById('refusal-message');
      answerTextEl.textContent = 'default text';
      refusalMessageEl.style.display = 'none';

      await runAsk(payload);

      // Verify that refusal message is shown and answer text is not rendered (cleared/empty)
      expect(refusalMessageEl.style.display).toBe('block');
      expect(answerTextEl.textContent).toBe('');
    }
  });
});
