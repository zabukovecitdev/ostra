LIMIT ?= 5

.DEFAULT_GOAL := help

.PHONY: help run reindex search test lint format check

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

run: ## Process the configured files
	uv run personal-rag

reindex: ## Process every file again, including chunks already stored
	uv run personal-rag --reindex

search: ## Print the stored chunks closest to QUERY, e.g. make search QUERY="solar system" LIMIT=3
	@test -n "$(QUERY)" || { echo 'Set QUERY, e.g. make search QUERY="solar system"'; exit 1; }
	uv run personal-rag --search "$(QUERY)" --limit $(LIMIT)

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
