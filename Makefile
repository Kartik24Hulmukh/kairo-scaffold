# Makefile for Kairo-Scaffold Monorepo

ifeq ($(OS),Windows_NT)
    VENV_DIR = kernel/sidecar/.venv
    VENV_PYTHON = $(VENV_DIR)/Scripts/python.exe
    VENV_PYTEST = $(VENV_DIR)/Scripts/pytest.exe
    NPM = npm.cmd
else
    VENV_DIR = kernel/sidecar/.venv
    VENV_PYTHON = $(VENV_DIR)/bin/python
    VENV_PYTEST = $(VENV_DIR)/bin/pytest
    NPM = npm
endif

.PHONY: build test demo bench gauntlet safety acceptance domains-check clean license-check setup-venv eval perf not-list-check receipt-check overlay-test

build: setup-venv
	cargo build
	python -c "import glob, py_compile; [py_compile.compile(f) for f in glob.glob('kernel/sidecar/**/*.py', recursive=True) + glob.glob('cli/**/*.py', recursive=True) + glob.glob('packs/**/*.py', recursive=True)]"

overlay-test: setup-venv
	$(VENV_PYTEST) kernel/tests/test_overlay_platform.py -v

setup-venv:
	python scripts/setup_venv.py

test: setup-venv
	cargo test
	$(VENV_PYTEST) kernel/tests/ packs/tests/

demo:
	cargo run --bin kairo -- run fixtures/golden/placeholder.txt --pack generic

bench:
	$(VENV_PYTHON) bench/run_bench.py

# Generate adversarial gauntlet fixtures then run full benchmark over them
gauntlet: setup-venv
	@echo "==> Generating adversarial gauntlet fixtures..."
	$(VENV_PYTHON) scripts/generate_gauntlet_fixtures.py
	@echo "==> Running full benchmark (includes adversarial fixtures)..."
	$(VENV_PYTHON) bench/run_bench.py
	@echo "==> Gauntlet complete. See bench/leaderboard.html for results."

safety:
	cargo check
	$(VENV_PYTHON) -m pytest --version

# Full acceptance pipeline: unit tests + all 8 premortem gates + benchmark hard gates
# A1: Grounding gauntlet (rotated, multi-col, dense table, low-DPI, non-English)
# A2: Perf CI budget enforcement
# A3: Near-miss refusal threshold (<5% false-refusal)
# A4: License contamination guard (no AGPL/BSL in MIT core)
# A5: Adversarial bbox suite (B3/D2 verifier blocks)
# A6: SPEC §9 NOT-list enforcement
# A7: RAGShield KB poisoning defense
# A8: Click-to-source citation coverage
acceptance: test perf not-list-check license-check overlay-test
	@echo "==> Running grounding benchmark (make bench)..."
	$(VENV_PYTHON) bench/run_bench.py
	@echo "==> Checking SPEC §5 hard gates..."
	$(VENV_PYTHON) scripts/run_acceptance.py
	@echo "==> [A1] Grounding gauntlet: rotated, multi-col, dense table, low-DPI, non-English..."
	$(VENV_PYTEST) kernel/tests/test_acceptance_gauntlet.py -v
	@echo "==> [A2] Perf budget CI enforcement..."
	$(VENV_PYTEST) kernel/tests/test_perf_ci.py -v
	@echo "==> [A3] Near-miss refusal threshold (false-refusal <5%)..."
	$(VENV_PYTEST) kernel/tests/test_near_miss_refusal.py -v
	@echo "==> [A4] License contamination guard (no AGPL/BSL in core)..."
	$(VENV_PYTEST) kernel/tests/test_license_guard.py -v
	@echo "==> [A5] Adversarial bbox suite (15 cases, VGVA blocks hallucinated boxes)..."
	$(VENV_PYTEST) kernel/tests/test_vgva_verifier.py -v
	@echo "==> [A6] SPEC §9 NOT-list enforcement..."
	$(VENV_PYTEST) kernel/tests/test_not_list_ci.py -v
	@echo "==> [A7] RAGShield KB poisoning defense (10 vectors)..."
	$(VENV_PYTEST) kernel/tests/test_rag_shield_integration.py -v
	@echo "==> [A8] Click-to-source citation coverage..."
	$(VENV_PYTEST) kernel/tests/test_source_coverage.py -v
	@echo ""
	@echo "============================================"
	@echo "  ALL 8 PREMORTEM ACCEPTANCE GATES PASSED  "
	@echo "============================================"

domains-check: setup-venv
	$(VENV_PYTEST) packs/tests/

license-check:
	@echo "Checking license compliance..."
	python scripts/ci/license_check.py

clean:
	cargo clean
	python -c "import shutil, os; shutil.rmtree('kernel/sidecar/.venv') if os.path.exists('kernel/sidecar/.venv') else None; shutil.rmtree('overlay/node_modules') if os.path.exists('overlay/node_modules') else None; shutil.rmtree('overlay/dist') if os.path.exists('overlay/dist') else None"

# Eval harness (C4): runs Ragas/DeepEval-style grounding metrics, appends to bench/history.jsonl
eval: setup-venv
	$(VENV_PYTHON) bench/eval_harness.py --inject-fabricated
	$(VENV_PYTHON) bench/check_regression.py

# Performance budget check (E1/C3): structural check + measured budgets when sidecar is running
perf: setup-venv
	$(VENV_PYTHON) scripts/check_perf_budget.py

# v1 NOT-list enforcement (E1/Fix6): fails if out-of-scope modules or forbidden phrases found
not-list-check:
	$(VENV_PYTHON) scripts/check_not_list.py

# Receipt-check (A1): ensures all tasks have a receipt in docs/receipts/
receipt-check:
	$(VENV_PYTHON) scripts/check_receipt.py

