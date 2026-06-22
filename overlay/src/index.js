import { invoke } from '@tauri-apps/api/core';

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  const docSelector = document.getElementById('doc-selector');
  const queryInput = document.getElementById('query-input');
  const askBtn = document.getElementById('ask-btn');
  const answerText = document.getElementById('answer-text');
  const refusalMessage = document.getElementById('refusal-message');
  const citationsContainer = document.getElementById('citations-container');
  const pageImage = document.getElementById('page-image');
  const highlightOverlay = document.getElementById('highlight-overlay');

  // Load documents on startup
  try {
    const docs = await invoke('get_documents');
    docs.forEach(doc => {
      const option = document.createElement('option');
      option.value = doc.doc_id;
      // Show file name or path
      const filename = doc.source_path.split('/').pop().split('\\').pop();
      option.textContent = `${filename} (${doc.doc_id})`;
      docSelector.appendChild(option);
    });
  } catch (err) {
    console.error("Failed to load documents:", err);
  }

  // Handle Ask button click
  askBtn.addEventListener('click', handleAsk);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      handleAsk();
    }
  });

  async function handleAsk() {
    const docId = docSelector.value;
    const query = queryInput.value.trim();

    if (!docId || !query) {
      alert("Please select a document and enter a query.");
      return;
    }

    // Reset previous state
    answerText.textContent = '';
    refusalMessage.style.display = 'none';
    citationsContainer.innerHTML = '';
    pageImage.style.display = 'none';
    highlightOverlay.innerHTML = '';

    try {
      const response = await fetch('http://127.0.0.1:7438/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ doc_id: docId, query })
      });

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`);
      }

      const data = await response.json();

      // Render answer ONLY IF grounded is true, text is not "blocked", and there is at least one citation
      if (data && data.grounded === true && data.text && data.text !== 'blocked' && data.citations && data.citations.length > 0) {
        answerText.textContent = data.text;

        data.citations.forEach((citation, index) => {
          const chip = document.createElement('span');
          chip.className = 'citation-chip';
          chip.textContent = `Citation ${index + 1} (Page ${citation.page})`;
          chip.addEventListener('click', () => showCitation(citation, docId));
          citationsContainer.appendChild(chip);
        });
      } else {
        // Render refusal message
        refusalMessage.style.display = 'block';
      }
    } catch (err) {
      console.error("Error asking sidecar:", err);
      refusalMessage.style.display = 'block';
    }
  }

  async function showCitation(citation, docId) {
    try {
      const metadata = await invoke('get_page_metadata', { docId, pageIndex: citation.page });
      const { width_px: dbWidth, height_px: dbHeight, image_sha256 } = metadata;

      highlightOverlay.innerHTML = '';
      pageImage.style.display = 'block';
      pageImage.src = `kairo-img://localhost/${image_sha256}.png`;

      const drawHighlight = () => {
        const renderedWidth = pageImage.clientWidth;
        const renderedHeight = pageImage.clientHeight;

        let x0 = citation.bbox.x0;
        let y0 = citation.bbox.y0;
        let x1 = citation.bbox.x1;
        let y1 = citation.bbox.y1;

        // Scale normalized coordinates if necessary
        if (x0 <= 1.0 && x1 <= 1.0 && y0 <= 1.0 && y1 <= 1.0) {
          x0 *= dbWidth;
          x1 *= dbWidth;
          y0 *= dbHeight;
          y1 *= dbHeight;
        }

        const scaleX = renderedWidth / dbWidth;
        const scaleY = renderedHeight / dbHeight;

        const x0_rendered = x0 * scaleX;
        const x1_rendered = x1 * scaleX;
        const y0_rendered = y0 * scaleY;
        const y1_rendered = y1 * scaleY;

        highlightOverlay.style.left = `${pageImage.offsetLeft}px`;
        highlightOverlay.style.top = `${pageImage.offsetTop}px`;
        highlightOverlay.style.width = `${pageImage.clientWidth}px`;
        highlightOverlay.style.height = `${pageImage.clientHeight}px`;

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

      pageImage.onload = drawHighlight;
      if (pageImage.complete) {
        drawHighlight();
      }
    } catch (err) {
      console.error("Failed to show citation page/image:", err);
    }
  }
}
