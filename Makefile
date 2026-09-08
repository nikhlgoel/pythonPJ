PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv install test lint fmt type bench serve demo clean

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install -e ".[all]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests

type:
	$(PY) -m mypy

bench:
	$(PY) -m embra.bench.suite --n 20000 --dim 128 --queries 200

serve:
	$(PY) -m uvicorn embra.server.app:create_default_app --factory --host 0.0.0.0 --port 8080

demo:
	$(PY) examples/rag_pipeline.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build **/__pycache__
