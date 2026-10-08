from unittest.mock import MagicMock, patch

import psycopg
import pytest
from fastapi import HTTPException

from backend.app.database import check_database
from backend.app.main import database_health, health


def test_database_connection_is_closed(monkeypatch):
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    connection = MagicMock()
    with patch("backend.app.database.psycopg.connect") as connect:
        connect.return_value.__enter__.return_value = connection
        check_database()
        connection.execute.assert_called_once_with("SELECT 1")
        assert connect.return_value.__exit__.called
        assert connect.call_args.kwargs["connect_timeout"] == 3


def test_database_readiness_success():
    with patch("backend.app.main.check_database"):
        assert database_health() == {"status": "healthy", "database": "connected"}


@pytest.mark.parametrize(
    "error", [psycopg.OperationalError("secret"), RuntimeError("secret")]
)
def test_database_readiness_failure_does_not_leak_details(error):
    with (
        patch("backend.app.main.check_database", side_effect=error),
        pytest.raises(HTTPException) as caught,
    ):
        database_health()
    assert caught.value.status_code == 503
    assert caught.value.detail == "Database unavailable"
    assert health() == {"status": "healthy"}


def test_missing_password_fails_before_connecting(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with patch("backend.app.database.psycopg.connect") as connect:
        with pytest.raises(RuntimeError, match="not configured"):
            check_database()
        connect.assert_not_called()
