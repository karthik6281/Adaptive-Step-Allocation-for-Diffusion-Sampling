.PHONY: check tables latex figures lint clean

check:
	python scripts/check.py

tables:
	python scripts/tables.py

latex:
	python scripts/tables.py --latex

figures:
	python scripts/figures.py --out build

lint:
	ruff check src scripts

clean:
	rm -rf build .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
