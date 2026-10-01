# ==============================================================================
# Makefile for UTH Computer Vision Workspace
# ==============================================================================

.PHONY: help install install-dev install-all format lint type-check test test-smoke clean run-lab setup-hooks

SHELL := /bin/bash
PYTHON := python3

help:
	@echo "UTH Computer Vision Workspace Automation Commands:"
	@echo "  make install        Install base & deep learning dependencies"
	@echo "  make install-dev    Install developer tools & tracking dependencies"
	@echo "  make install-all    Install editable package with all optional extras"
	@echo "  make setup-hooks    Install pre-commit hooks into git repository"
	@echo "  make format         Auto-format code using Ruff and Black"
	@echo "  make lint           Check code quality and style using Ruff"
	@echo "  make type-check     Run static type checker with Mypy"
	@echo "  make test           Run all unit tests with pytest"
	@echo "  make test-smoke     Run quick sanity smoke tests with pytest"
	@echo "  make clean          Clean temporary files, caches, and build artifacts"
	@echo "  make run-lab        Run lab script helper (e.g. LAB=1 make run-lab)"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements/dev.txt -r requirements/tracking.txt

install-all:
	$(PYTHON) -m pip install -e .[all]

setup-hooks:
	pre-commit install

format:
	ruff format src tests
	ruff check --fix src tests
	black src tests

lint:
	ruff check src tests
	black --check src tests

type-check:
	mypy src tests

test:
	pytest tests/ -v

test-smoke:
	pytest tests/ -m smoke -v

run-lab:
	@bash scripts/run_lab.sh $(LAB)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*.so" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Workspace cleaned successfully!"
