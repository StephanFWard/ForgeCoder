# ForgeCoder — Makefile
# Targets are designed to work on Windows (cmd/PowerShell-friendly) and POSIX.
# On Windows you may need to `pip install make` (GnuWin32) or use the
# equivalent cmd/PowerShell commands documented in README.md.

PYTHON       := python
PIP          := $(PYTHON) -m pip
VENV         := .venv
EXT_DIR      := apps/vscode

.PHONY: help venv install dev test test-py test-vs lint serve llama tower \
	build-vs package-vs clean

help:
	@echo "ForgeCoder targets:"
	@echo "  venv        Create a virtualenv (.venv)"
	@echo "  install     pip install -e . (runtime deps)"
	@echo "  dev         pip install -e \".[dev]\" (runtime + dev deps)"
	@echo "  test        Run Python unit + integration tests"
	@echo "  lint        Run ruff over Python code"
	@echo "  serve       Start the Forge server on 127.0.0.1:8787"
	@echo "  llama       Start llama-server on 127.0.0.1:8080 (model required)"
	@echo "  build-vs    Compile the VS Code extension (TypeScript)"
	@echo "  package-vs  Build apps/vscode/*.vsix"
	@echo "  clean       Remove build caches"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

test: test-py

test-py:
	$(PYTHON) -m pytest

test-vs:
	cd $(EXT_DIR) && npm test

lint:
	$(PYTHON) -m ruff check apps/server core tests training/scripts

serve:
	$(PYTHON) -m uvicorn forge_server.main:app --host 127.0.0.1 --port 8787

llama:
	$(PYTHON) runtime/scripts/launch_llama.py

build-vs:
	cd $(EXT_DIR) && npm install && npm run compile

package-vs:
	cd $(EXT_DIR) && npm install && npm run compile && npm run package

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info ./**/__pycache__