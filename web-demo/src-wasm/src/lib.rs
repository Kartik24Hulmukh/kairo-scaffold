use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::cell::RefCell;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Citation {
    #[serde(default)]
    pub chunk_id: Option<String>,
    pub page: u32,
    pub bbox: serde_json::Value,
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct QueryResult {
    pub grounded: bool,
    pub answer: String,
    pub citations: Vec<Citation>,
}

// WASM is single-threaded; thread_local + RefCell is the idiomatic safe replacement
// for static mut in wasm-bindgen contexts (no Send/Sync required).
thread_local! {
    static DOCUMENTS: RefCell<HashMap<String, Vec<Citation>>> = RefCell::new(HashMap::new());
}

#[wasm_bindgen]
pub fn clear_engine() {
    DOCUMENTS.with(|docs| docs.borrow_mut().clear());
}

#[wasm_bindgen]
pub fn init_engine() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn index_document(doc_id: &str, chunks_json: &str) -> bool {
    let chunks: Result<Vec<Citation>, _> = serde_json::from_str(chunks_json);
    match chunks {
        Ok(c) => {
            DOCUMENTS.with(|docs| docs.borrow_mut().insert(doc_id.to_string(), c));
            true
        }
        Err(e) => {
            let _ = e;
            false
        }
    }
}

#[wasm_bindgen]
pub fn query_engine(doc_id: &str, query: &str) -> String {
    // Clone the chunks out of the RefCell so we don't hold a borrow across
    // the subsequent computation (no nested borrow issues).
    let chunks_opt: Option<Vec<Citation>> =
        DOCUMENTS.with(|docs| docs.borrow().get(doc_id).cloned());
    let chunks = match chunks_opt {
        Some(c) => c,
        None => {
            let res = QueryResult {
                grounded: false,
                answer: "Document not indexed.".to_string(),
                citations: vec![],
            };
            return serde_json::to_string(&res).unwrap_or_default();
        }
    };

    if query.trim().is_empty() {
        let res = QueryResult {
            grounded: false,
            answer: "Empty query.".to_string(),
            citations: vec![],
        };
        return serde_json::to_string(&res).unwrap_or_default();
    }

    let query_terms: Vec<String> = query
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();

    if query_terms.is_empty() {
        let res = QueryResult {
            grounded: false,
            answer: "I am sorry, but the query could not be answered from the document source.".to_string(),
            citations: vec![],
        };
        return serde_json::to_string(&res).unwrap_or_default();
    }

    // Compute document frequencies for terms across this document's chunks.
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for term in &query_terms {
        let count = chunks.iter().filter(|chunk| chunk.text.to_lowercase().contains(term.as_str())).count();
        doc_freq.insert(term.clone(), count);
    }

    let num_chunks = chunks.len() as f64;
    let mut chunk_scores: Vec<(usize, f64)> = Vec::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        let chunk_text_lower = chunk.text.to_lowercase();
        let mut score = 0.0;
        for term in &query_terms {
            if chunk_text_lower.contains(term.as_str()) {
                let tf = chunk_text_lower.matches(term.as_str()).count() as f64;
                let df = *doc_freq.get(term).unwrap_or(&0) as f64;
                let idf = ((num_chunks + 1.0) / (df + 1.0)).ln() + 1.0;
                score += tf * idf;
            }
        }
        if score > 0.0 {
            chunk_scores.push((idx, score));
        }
    }

    // Sort descending by score.
    chunk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if chunk_scores.is_empty() {
        let res = QueryResult {
            grounded: false,
            answer: "I am sorry, but the query could not be answered from the document source.".to_string(),
            citations: vec![],
        };
        return serde_json::to_string(&res).unwrap_or_default();
    }

    let mut citations = Vec::new();
    for &(idx, _) in chunk_scores.iter().take(3) {
        citations.push(chunks[idx].clone());
    }

    let answer = format!("Found relevant match for query: {}. Best reference: {}", query, citations[0].text);

    let result = QueryResult {
        grounded: true,
        answer,
        citations,
    };

    serde_json::to_string(&result).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_citation(text: &str) -> Citation {
        Citation {
            chunk_id: Some("c1".to_string()),
            page: 1,
            bbox: serde_json::Value::Null,
            text: text.to_string(),
        }
    }

    #[test]
    fn test_query_unindexed_doc_returns_not_grounded() {
        // A doc that was never indexed should return grounded=false.
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
        let result_json = query_engine("nonexistent-doc", "some query");
        let result: QueryResult = serde_json::from_str(&result_json).unwrap();
        assert!(!result.grounded);
        assert!(result.citations.is_empty());
    }

    #[test]
    fn test_query_empty_query_returns_not_grounded() {
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
        // Index a doc with some content.
        let chunks = vec![make_citation("This is revenue data for Q3.")];
        let chunks_json = serde_json::to_string(&chunks).unwrap();
        DOCUMENTS.with(|docs| {
            docs.borrow_mut().insert("doc1".to_string(), chunks.clone());
        });
        let _ = chunks_json;

        let result_json = query_engine("doc1", "   ");
        let result: QueryResult = serde_json::from_str(&result_json).unwrap();
        assert!(!result.grounded);
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
    }

    #[test]
    fn test_query_matching_term_returns_grounded() {
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
        let chunks = vec![
            make_citation("The quarterly revenue was $4.5 million."),
            make_citation("Operating costs increased by 12%."),
        ];
        DOCUMENTS.with(|docs| {
            docs.borrow_mut().insert("doc2".to_string(), chunks);
        });

        let result_json = query_engine("doc2", "quarterly revenue");
        let result: QueryResult = serde_json::from_str(&result_json).unwrap();
        assert!(result.grounded);
        assert!(!result.citations.is_empty());
        assert!(result.citations[0].text.contains("revenue"));
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
    }

    #[test]
    fn test_query_no_matching_term_returns_not_grounded() {
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
        let chunks = vec![make_citation("The weather is sunny today.")];
        DOCUMENTS.with(|docs| {
            docs.borrow_mut().insert("doc3".to_string(), chunks);
        });

        let result_json = query_engine("doc3", "revenue profit margin");
        let result: QueryResult = serde_json::from_str(&result_json).unwrap();
        assert!(!result.grounded);
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
    }

    #[test]
    fn test_index_invalid_json_returns_false() {
        let ok = index_document("doc_bad", "not valid json {{{{");
        assert!(!ok);
    }

    #[test]
    fn test_index_valid_json_returns_true() {
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
        let chunks = vec![make_citation("Test content.")];
        let chunks_json = serde_json::to_string(&chunks).unwrap();
        let ok = index_document("doc_good", &chunks_json);
        assert!(ok);
        DOCUMENTS.with(|docs| docs.borrow_mut().clear());
    }
}
