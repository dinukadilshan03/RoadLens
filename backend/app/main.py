import psycopg
from fastapi import FastAPI, HTTPException

from backend.app.database import check_database

app = FastAPI(
    title="RoadLens API",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/health/db")
def database_health():
    try:
        check_database()
    except (psycopg.Error, RuntimeError):
        # Do not expose connection details or credentials in the API response.
        raise HTTPException(status_code=503, detail="Database unavailable") from None
    return {"status": "healthy", "database": "connected"}
