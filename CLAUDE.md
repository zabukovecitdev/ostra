# Code complexity

Keep functions simple. Both checks are configured in `pyproject.toml`:

- Cognitive complexity (complexipy), max 15 per function: `uv run complexipy`
- Cyclomatic complexity (ruff C901 / mccabe), max 10: `uv run ruff check`

When writing or changing code, run both on the touched files and keep every function under the limits.
Aim well below them (cognitive ≤ 8 is a good target). Reduce complexity with early returns, small
helper functions and flatter control flow. Do not use `# noqa` or `# complexipy: ignore` to silence them.
Show the worst offenders with `uv run complexipy --sort desc --top 10`.
