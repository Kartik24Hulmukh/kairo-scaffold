# E3 — VGVA: Visual Grounding Verification Agent (Model-Independent)

**Task ID:** E3  
**Title:** VGVA — Formalized as dedicated model-independent verifier component  
**Status:** PASS  
**Date:** 2026-06-20  
**Commit SHA:** (working tree)

---

## PLAN

- Upgrade `kernel/sidecar/models/vgva.py` from a forward stub to a fully functional, model-independent text-presence verifier.
- Architecture:
  1. `verify(image_bytes, text_claim, bbox)`: Crop region → OCR (pytesseract) → fuzzy text match.
  2. `verify_text_present(page_text, text_claim)`: Direct fuzzy match against pre-extracted text.
  3. Both paths use `SequenceMatcher` (Ratcliff/Obershelp) + sliding-window substring search.
  4. Zero LLM calls — purely deterministic OCR + text matching.
  5. Safe fallback: when PIL absent → `{verified: False, method: "not_available"}`.
- Add `_fuzzy_match_ratio()` and `_normalise()` helpers as testable public functions.
- Add `verify_text_present()` as the fast path for text-only PDFs.
- Cover with `kernel/tests/test_vgva_verifier.py`: schema, types, model-independence, 100% block rate, 0% false positive rate, PIL fallback, helper unit tests.

**Gate command:** `pytest kernel/tests/test_vgva_verifier.py -v`

---

## CRITIQUE

- OCR accuracy depends on pytesseract + Tesseract binary installation. In CI without Tesseract, `verify()` falls back to `method="ocr_fallback_empty"` with `verified=False` — which is safe (conservative) but means the image path cannot verify true references in CI. The `verify_text_present()` path always works and is the recommended API for text-layer PDFs.
- SequenceMatcher is O(n·m) where n=|claim|, m=|text|. For very long page texts (>10k chars), sliding-window search adds O(n·m/step) — acceptable for single-page texts.
- The default threshold (0.70) is intentionally permissive to handle OCR noise (character transpositions, spacing). For high-security applications, raise to 0.85–0.90.
- The D-series `test_d_series.py::TestVisualGroundingVerificationAgent::test_verify_fallback_when_deps_absent` was updated to accept the new method names (`"ocr_pytesseract"`, `"ocr_fallback_empty"`) in addition to the legacy stub values.

---

## FILES CHANGED

- `kernel/sidecar/models/vgva.py` — complete rewrite: OCR+fuzzy verifier (~180 lines).
- `kernel/tests/test_vgva_verifier.py` — new test file (32 tests).
- `kernel/tests/test_d_series.py` — updated `test_verify_fallback_when_deps_absent` to accept upgraded method names.

---

## GATE COMMAND

```
pytest kernel/tests/test_vgva_verifier.py -v
```

---

## GATE OUTPUT (verbatim, real)

```
============================= test session starts =============================
platform win32 -- Python 3.12.0, pytest-9.0.3

kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_returns_dict PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_has_verified_key PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_has_confidence_key PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_has_method_key PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_has_ocr_text_key PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_text_present_returns_dict PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_text_present_has_all_keys PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAReturnSchema::test_verify_text_present_method_is_text_match PASSED
kernel/tests/test_vgva_verifier.py::TestVGVATypes::test_verified_is_bool PASSED
kernel/tests/test_vgva_verifier.py::TestVGVATypes::test_confidence_is_float PASSED
kernel/tests/test_vgva_verifier.py::TestVGVATypes::test_method_is_str PASSED
kernel/tests/test_vgva_verifier.py::TestVGVATypes::test_confidence_in_range PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAModelIndependence::test_method_is_not_llm PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAModelIndependence::test_method_is_not_llm_for_image_verify PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAFalseReferenceBlocking::test_not_present_cases_all_blocked PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAFalseReferenceBlocking::test_completely_unrelated_text_is_blocked PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAValidReferenceAcceptance::test_valid_cases_all_accepted PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAFallback::test_verify_returns_not_available_when_pil_missing PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_exact_containment_returns_1 PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_empty_claim_returns_0 PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_empty_text_returns_0 PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_completely_unrelated_is_low PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_exact_match_full_string PASSED
kernel/tests/test_vgva_verifier.py::TestFuzzyMatchRatio::test_near_match_is_high PASSED
kernel/tests/test_vgva_verifier.py::TestNormalise::test_lowercase PASSED
kernel/tests/test_vgva_verifier.py::TestNormalise::test_strips_punctuation PASSED
kernel/tests/test_vgva_verifier.py::TestNormalise::test_collapses_whitespace PASSED
kernel/tests/test_vgva_verifier.py::TestNormalise::test_empty_string PASSED
kernel/tests/test_vgva_verifier.py::TestNormalise::test_numbers_preserved PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAThreshold::test_strict_threshold_blocks_near_match PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAThreshold::test_permissive_threshold_accepts_paraphrase PASSED
kernel/tests/test_vgva_verifier.py::TestVGVAThreshold::test_default_threshold_is_sane PASSED

32 passed, 1 warning in 1.23s
```

---

## KEY GATES MET

| Gate | Result |
|------|--------|
| Block 100% of "reference not present" synthetic cases | ✅ PASS (5/5 blocked) |
| Accept 100% of valid verbatim references | ✅ PASS (5/5 accepted) |
| No LLM calls (model-independent) | ✅ PASS (method ≠ "llm_*") |
| PIL-absent safe fallback | ✅ PASS (not_available, verified=False) |
| Return schema (4 required keys) | ✅ PASS |
| Confidence in [0.0, 1.0] | ✅ PASS |

---

## NOTES

- VGVA is designed to be called by the RAG cascade AFTER retrieval, BEFORE answer generation, as an independent gate.
- The cascade should treat `verified=False` as a BLOCK signal, preventing hallucinated citations from surfacing in the answer.
- For production use with scanned documents, install Tesseract OCR and `pytesseract`. For text-layer PDFs, use `verify_text_present()` directly.
- Reference: ViG-LLM / VGVA (Amazon, 2026) — the concept of a model-independent visual grounding verification step called by the cascade without self-certification.
