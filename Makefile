.PHONY: help linter formatter security typecheck tests all clean

help:
	@echo "Available commands:"
	@echo "  make linter     - Run ruff linter checks"
	@echo "  make formatter  - Format code with black and ruff"
	@echo "  make security   - Run security checks with pysentry"
	@echo "  make typecheck  - Run mypy type checks"
	@echo "  make tests      - Run pytest tests"
	@echo "  make all        - Run formatter, linter, typecheck, security, and tests"
	@echo "  make clean      - Remove Python cache files"

linter:
	@echo "Running ruff linter..."
	uv run ruff check src/ tests/

formatter:
	@echo "Running black formatter..."
	uv run black src/ tests/
	@echo "Running ruff formatter..."
	uv run ruff format src/ tests/

typecheck:
	@echo "Running mypy type checks..."
	uv run mypy src/ tests/

security:
	@echo "Running pysentry security checks..."
	uv run python -c "import pysentry; pysentry.run_cli(['src/'])"

tests:
	@echo "Running pytest..."
	uv run pytest tests/ -v

all: formatter linter typecheck security tests
	@echo "All checks completed!"
