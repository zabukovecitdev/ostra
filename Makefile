LIMIT ?= 5
# Optional folder to process instead of the configured folders, e.g. make run DIRECTORY=~/Documents/wikipedia
DIRECTORY ?=
DIRECTORY_FLAG = $(if $(DIRECTORY),--directory $(DIRECTORY))

.DEFAULT_GOAL := help

.PHONY: help run reindex search eval eval-sample test lint format typecheck check hooks

help: ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

run: ## Index the configured files, or DIRECTORY when set
	uv run ostra index $(DIRECTORY_FLAG)

reindex: ## Embed every chunk again, or DIRECTORY when set
	uv run ostra index --reindex $(DIRECTORY_FLAG)

search: ## Print the stored chunks closest to QUERY, e.g. make search QUERY="solar system" LIMIT=3
	@test -n "$(QUERY)" || { echo 'Set QUERY, e.g. make search QUERY="solar system"'; exit 1; }
	uv run ostra search "$(QUERY)" --limit $(LIMIT)

eval: ## Measure retrieval on CASES (a JSON test set), e.g. make eval CASES=evals/my_cases.json LIMIT=5
	@test -n "$(CASES)" || { echo 'Set CASES, e.g. make eval CASES=evals/my_cases.json'; exit 1; }
	uv run ostra eval "$(CASES)" --limit $(LIMIT) $(if $(QUERY_PREFIX),--query-prefix "$(QUERY_PREFIX)")

# The sample run uses its own HOME, so it never touches your config or database. It needs a running Ollama.
SAMPLE_HOME = .eval-sample
eval-sample: ## Index tests/data into a scratch database and measure retrieval on evals/sample_cases.json
	mkdir -p $(SAMPLE_HOME)/ostra
	printf '{"database_path": "%s/ostra/ostra.db", "directories": ["%s/tests/data"], "chunk_size_tokens": 128, "chunk_overlap_tokens": 16}' "$(CURDIR)/$(SAMPLE_HOME)" "$(CURDIR)" > $(SAMPLE_HOME)/ostra/config.json
	uv run --no-sync env HOME="$(CURDIR)/$(SAMPLE_HOME)" ostra index
	uv run --no-sync env HOME="$(CURDIR)/$(SAMPLE_HOME)" ostra eval evals/sample_cases.json --limit $(LIMIT) $(if $(QUERY_PREFIX),--query-prefix "$(QUERY_PREFIX)")

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
