LIMIT ?= 5
# Optional folder to process instead of the configured folders, e.g. make run DIRECTORY=~/Documents/wikipedia
DIRECTORY ?=
DIRECTORY_FLAG = $(if $(DIRECTORY),--directory $(DIRECTORY))
# Set RECURSIVE=1 to index the files in subfolders too, e.g. make run DIRECTORY=~/notes RECURSIVE=1
RECURSIVE ?=
RECURSIVE_FLAG = $(if $(RECURSIVE),--recursive)

.DEFAULT_GOAL := help

.PHONY: help install uninstall run reindex search test lint format typecheck check hooks

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install the ostra commands for your user
	uv tool install --editable --reinstall .

uninstall: ## Remove the installed ostra commands
	uv tool uninstall ostra

run: ## Index the configured files, or DIRECTORY when set. RECURSIVE=1 scans subfolders
	uv run ostra index $(DIRECTORY_FLAG) $(RECURSIVE_FLAG)

reindex: ## Embed every chunk again, or DIRECTORY when set
	uv run ostra index --reindex $(DIRECTORY_FLAG) $(RECURSIVE_FLAG)

search: ## Print the stored chunks closest to QUERY, e.g. make search QUERY="solar system" LIMIT=3
	@test -n "$(QUERY)" || { echo 'Set QUERY, e.g. make search QUERY="solar system"'; exit 1; }
	uv run ostra search "$(QUERY)" --limit $(LIMIT)

test: ## Run the test suite
	uv run pytest

lint: ## Check the code with ruff
	uv run ruff check .

format: ## Format the code with ruff
	uv run ruff format .

typecheck: ## Check the types with mypy
	uv run mypy

check: ## Run the lint, format check, type check and tests
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run pytest

hooks: ## Install the git hooks (checks on commit, tests on push)
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push
