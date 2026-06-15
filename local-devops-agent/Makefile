# Configuration - Update these or override via environment variables
MODEL_PATH ?= google/gemma-4-E2B-it

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.log" -exec rm -rf {} +

.PHONY: install run test clean lint help

help:
	@echo "Available Makefile commands:"
	@echo ""
	@echo "  install   - Installs dependencies from requirements.txt"
	@echo "  run       - Runs the local SRE agent MCP server"
	@echo "  test      - Runs the unit test suite"
	@echo "  lint      - Runs Ruff checks, Ruff formatting checks, and MyPy typechecking"
	@echo "  clean     - Cleans up caches and temporary files"
	@echo ""
