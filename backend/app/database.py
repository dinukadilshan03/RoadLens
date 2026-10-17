"""PostgreSQL connection settings and the readiness check for the local demo."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Environment supplied by Compose takes precedence over the local .env file.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def connection_params() -> dict:
    """Connection keyword arguments for psycopg.connect, from the environment."""
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is not configured")
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": os.environ.get("POSTGRES_PORT", "5432"),
        "dbname": os.environ.get("POSTGRES_DB", "roadlens"),
        "user": os.environ.get("POSTGRES_USER", "roadlens"),
        "password": password,
    }


def check_database() -> None:
    """Check authentication and query execution; always close the connection."""
    with psycopg.connect(
        **connection_params(),
        connect_timeout=3,
        options="-c statement_timeout=3000",
    ) as connection:
        connection.execute("SELECT 1").fetchone()
