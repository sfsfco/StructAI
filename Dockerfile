# ============================================================
# Stage 1 — Build: install Python deps in an isolated layer
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile native wheels (FAISS, asyncpg, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2 — Runtime: slim image with only what we need
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="StructAI Team" \
      description="StructAI API – FastAPI application" \
      version="0.1.0"

# Minimal runtime libs (libpq for asyncpg, curl for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Create FAISS data directory and grant ownership
RUN mkdir -p /app/data/faiss && chown -R appuser:appuser /app/data

# Drop privileges
USER appuser

EXPOSE 8000

# Container-level healthcheck (Docker & Compose)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["gunicorn", "app.main:app", "-c", "app/core/gunicorn_conf.py", "--bind", "0.0.0.0:8000"]
