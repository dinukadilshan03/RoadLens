FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for container logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install system dependencies required by OpenCV (video/image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a non-privileged system user and establish application directory ownership
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Copy dependency specifications together to maximize caching
COPY pyproject.toml uv.lock ./
COPY ml/c4/requirements.lock ./ml/c4/requirements.lock

# Install dependencies into /app/.venv with persistent BuildKit cache and no redundant layers
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    uv pip install --python /app/.venv/bin/python -r ml/c4/requirements.lock

# Copy application packages directly with appuser ownership (avoids slow recursive chown on .venv)
COPY --chown=appuser:appuser backend ./backend
COPY --chown=appuser:appuser ml ./ml
COPY --chown=appuser:appuser shared ./shared

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
