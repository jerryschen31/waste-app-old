.PHONY: setup lint test format verify

SWIFT?=swift
PYTHON?=python3

setup:
	@$(PYTHON) -m pip install --upgrade pip coremltools >/dev/null 2>&1 || echo "[setup] Skipping coremltools install (check environment)."

lint:
	@if command -v swiftlint >/dev/null 2>&1; then \
		swiftlint --strict; \
	else \
		echo "[lint] swiftlint not installed; skipping."; \
	fi

format:
	@if command -v swiftformat >/dev/null 2>&1; then \
		swiftformat .; \
	else \
		echo "[format] swiftformat not installed; skipping."; \
	fi

verify:
	@./scripts/verify-coreml.sh

test:
	@$(SWIFT) test --parallel
