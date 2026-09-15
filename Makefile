.PHONY: help linter formatter security typecheck tests all clean corpus-dry-run

help:
	@echo "Available commands:"
	@echo "  make linter     - Run ruff linter checks"
	@echo "  make formatter  - Format code with black and ruff"
	@echo "  make security   - Run security checks with pysentry"
	@echo "  make typecheck  - Run mypy type checks"
	@echo "  make tests      - Run pytest tests"
	@echo "  make all        - Run formatter, linter, typecheck, security, and tests"
	@echo "  make clean      - Remove Python cache files"
	@echo "  make corpus-dry-run - Chunk report for the oreo-data corpus (no services needed)"

linter:
	@echo "Running ruff linter..."
	uv run ruff check src/ tests/ scripts/

formatter:
	@echo "Running black formatter..."
	uv run black src/ tests/ scripts/
	@echo "Running ruff formatter..."
	uv run ruff format src/ tests/ scripts/

typecheck:
	@echo "Running mypy type checks..."
	uv run mypy src/ tests/ scripts/

security:
	@echo "Running pysentry security checks..."
	uv run python -c "import pysentry; pysentry.run_cli(['src/'])"

tests:
	@echo "Running pytest..."
	uv run pytest tests/ -v

corpus-dry-run:
	uv run python -m scripts.load_corpus --dry-run

all: formatter linter typecheck security tests
	@echo "All checks completed!"
