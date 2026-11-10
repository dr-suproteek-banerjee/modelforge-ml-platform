FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
ARG ENABLE_TORCH=0
RUN if [ "$ENABLE_TORCH" = "1" ]; then uv sync --locked --no-dev --extra torch; else uv sync --locked --no-dev; fi
COPY modelforge modelforge
COPY alembic.ini .
COPY migrations migrations
RUN useradd --uid 10001 --create-home modelforge && mkdir -p /artifacts && chown -R modelforge:modelforge /artifacts
USER modelforge
EXPOSE 8000
CMD ["uvicorn", "modelforge.api:app", "--host", "0.0.0.0", "--port", "8000"]
