FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for container logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install uv using the official standalone binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create a non-privileged system user for container security
RUN useradd -m -u 1000 appuser

# Install locked dependencies into /app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application packages (ML datasets, runs, and weights excluded via .dockerignore)
COPY backend ./backend
COPY ml ./ml
COPY shared ./shared

# Set ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]