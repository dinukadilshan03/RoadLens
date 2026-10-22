FROM python:3.11-slim

# Prevent Python from writing .pyc files at runtime, pre-compile bytecode during uv sync,
# and use copy link mode so BuildKit cache mounts work across filesystems.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install minimal system dependencies required by headless OpenCV and OpenMP (xgboost-cpu / torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a non-privileged system user and establish application directory ownership
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Copy unified dependency specifications to maximize Docker layer caching
COPY pyproject.toml uv.lock ./

# Install runtime-only dependencies into /app/.venv in a single deterministic pass
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy application packages directly with appuser ownership (avoids slow recursive chown on .venv)
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser ml ./ml
COPY --chown=appuser:appuser shared ./shared

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
