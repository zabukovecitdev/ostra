LIMIT ?= 5
# Optional folder to process instead of the configured folders, e.g. make run DIRECTORY=~/Documents/wikipedia
DIRECTORY ?=
DIRECTORY_FLAG = $(if $(DIRECTORY),--directory $(DIRECTORY))

.DEFAULT_GOAL := help

.PHONY: help run reindex search test lint format check

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

run: ## Index the configured files, or DIRECTORY when set
	uv run ostra index $(DIRECTORY_FLAG)

reindex: ## Embed every chunk again, or DIRECTORY when set
	uv run ostra index --reindex $(DIRECTORY_FLAG)

search: ## Print the stored chunks closest to QUERY, e.g. make search QUERY="solar system" LIMIT=3
	@test -n "$(QUERY)" || { echo 'Set QUERY, e.g. make search QUERY="solar system"'; exit 1; }
	uv run ostra search "$(QUERY)" --limit $(LIMIT)

test: ## Run the test suite
	uv run pytest

lint: ## Check the code with ruff
	uv run ruff check .

format: ## Format the code with ruff
	uv run ruff format .

check: ## Run the lint, format check and tests
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest
