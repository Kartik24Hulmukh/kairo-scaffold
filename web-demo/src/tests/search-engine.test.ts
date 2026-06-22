import { describe, it, expect, beforeAll } from 'vitest';
import { initSearchEngine, indexDocument, queryEngine } from '../search-engine';

describe('Search Engine Module Scaffold Tests', () => {
  beforeAll(async () => {
    await initSearchEngine();
  });

  it('should index document and return true', () => {
    const success = indexDocument('doc1', 'Hello this is Kairo 1000x Upgrade testing.');
    expect(success).toBe(true);
  });

  it('should query successfully for keywords', () => {
    indexDocument('doc1', 'Hello this is Kairo 1000x Upgrade testing.');
    const result = queryEngine('doc1', 'Kairo');
    expect(result.grounded).toBe(true);
    expect(result.citations.length).toBeGreaterThan(0);
    expect(result.citations[0].text).toContain('Kairo');
  });

  it('should refuse queries not found in the document', () => {
    indexDocument('doc1', 'Hello this is Kairo 1000x Upgrade testing.');
    const result = queryEngine('doc1', 'Unrelated query');
    expect(result.grounded).toBe(false);
    expect(result.answer).toContain('could not be answered');
  });
});
