install:
    uv sync

test:
    uv run pytest -q

lint:
    uv run ruff check .

fmt:
    uv run ruff format .
