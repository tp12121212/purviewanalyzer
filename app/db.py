from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_database_url


class Base(DeclarativeBase):
    pass


logger = logging.getLogger("presidio-streamlit")


def _sqlite_path_from_url(database_url: str) -> Path | None:
    if not database_url.startswith("sqlite:///"):
        return None
    path = database_url.replace("sqlite:///", "", 1)
    if not path or path == ":memory:":
        return None
    return Path(path)


def _ensure_sqlite_dir(database_url: str) -> None:
    db_path = _sqlite_path_from_url(database_url)
    if db_path is None:
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)


_database_url = get_database_url()
_ensure_sqlite_dir(_database_url)

engine = create_engine(
    _database_url,
    connect_args=(
        {"check_same_thread": False, "timeout": 30}
        if _database_url.startswith("sqlite")
        else {}
    ),
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
_db_init_lock = threading.Lock()
_db_initialized = False


if _database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
        except Exception:
            pass


def init_db() -> None:
    global _db_initialized
    from app import models  # noqa: F401

    if _db_initialized:
        return

    with _db_init_lock:
        if _db_initialized:
            return

        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                Base.metadata.create_all(bind=engine)
                _db_initialized = True
                return
            except OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
                last_exc = exc
                wait_seconds = min(0.25 * (attempt + 1), 1.5)
                logger.warning(
                    "SQLite database locked during init_db (attempt %s/6); retrying in %.2fs",
                    attempt + 1,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

        if last_exc:
            raise last_exc


def get_session() -> Generator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
