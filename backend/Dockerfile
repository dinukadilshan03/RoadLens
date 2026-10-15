FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Component 4 (road roughness) inference dependencies: scipy, scikit-learn, xgboost, ...
COPY ml/c4/requirements.lock ./ml/c4/requirements.lock
RUN uv pip install --no-cache --python /app/.venv/bin/python -r ml/c4/requirements.lock

COPY backend ./backend
COPY ml ./ml
COPY shared ./shared

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
