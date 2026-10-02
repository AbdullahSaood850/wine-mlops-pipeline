# GNU Makefile for Wine MLOps Pipeline Automation

PYTHON := python3
PIP := pip

.PHONY: help install lint test train clean

help:
	@echo "Available Makefile targets:"
	@echo "  install  : Upgrade pip and install all dependencies from requirements.txt"
	@echo "  lint     : Run flake8 across src/ and tests/ with max line length 100"
	@echo "  test     : Run all unit tests via pytest with verbose output"
	@echo "  train    : Execute MLflow training script with default arguments"
	@echo "  clean    : Remove compiled bytecode (*.pyc), test caches, and temporary files"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v tests/

train:
	$(PYTHON) -m src.train

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
