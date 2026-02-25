# ---------------------------------------------------------------------------
# Hermes Financial -- Multi-stage Docker build
#
# Stage 1 (builder): installs dependencies with uv into a virtual env.
# Stage 2 (runtime): copies the venv into a slim image with only the
#   runtime dependencies needed (including libreoffice for PDF export).
# ---------------------------------------------------------------------------

# ---- Stage 1: Builder ----------------------------------------------------
FROM python:3.12-slim AS builder

# Install uv -- the fast Python package installer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project metadata first for better layer caching.  If only source
# code changes (not dependencies) Docker can reuse the install layer.
COPY pyproject.toml ./
COPY hermes/ ./hermes/

# Create a virtual environment and install all dependencies.
# Using --system would install globally, but a venv is cleaner to copy.
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install .

# ---- Stage 2: Runtime ----------------------------------------------------
FROM python:3.12-slim AS runtime

# Install runtime system dependencies:
#   - libreoffice-writer-nogui: headless LibreOffice for docx-to-PDF export
#   - curl: health checks
# Clean up apt cache to keep the image small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libreoffice-writer-nogui \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc files and enable unbuffered output
# so that logs appear immediately in docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy the virtual environment from the builder stage.
COPY --from=builder /app/.venv /app/.venv

# Copy application source code.
COPY hermes/ ./hermes/
COPY examples/ ./examples/

# Ensure the venv's Python and scripts are on PATH.
ENV PATH="/app/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv"

# Create directories for runtime data.
RUN mkdir -p /app/data /app/cache /app/output

# Default environment variables (override with docker-compose or -e flags).
ENV HERMES_CHROMA_PERSIST_DIR=/app/data/chroma \
    HERMES_OUTPUT_DIR=/app/output \
    HERMES_CACHE_DIR=/app/cache

# Expose the default API port.
EXPOSE 8000

# Default command -- drop into a Python shell with hermes importable.
# Override in docker-compose.yml or at runtime for specific tasks.
CMD ["python", "-m", "hermes"]
