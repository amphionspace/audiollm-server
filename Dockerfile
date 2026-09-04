FROM ghcr.io/astral-sh/uv:0.8.22-python3.10-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libc++1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY backend ./backend
COPY frontend ./frontend
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
