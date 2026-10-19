import uuid

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.database import check_database
from backend.app.routes.c1 import router as c1_router
from backend.app.routes.c4 import router as c4_router

app = FastAPI(
    title="RoadLens API",
    version="0.1.0",
)

app.include_router(c1_router, prefix="/api/v1")
app.include_router(c4_router, prefix="/api/v1")


@app.exception_handler(psycopg.Error)
async def database_error(request: Request, exc: psycopg.Error):
    # Do not expose connection details or credentials in the API response.
    return JSONResponse(
        status_code=503,
        content={"code": "database_unavailable", "message": "Database unavailable",
                 "request_id": uuid.uuid4().hex, "details": {}},
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
