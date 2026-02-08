from __future__ import annotations

import logging
import os
import shutil
import sqlite3
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


def _sqlite_total_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cur.fetchall()]
        total = 0
        for table in tables:
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{table}"')
                total += int(cur.fetchone()[0])
            except Exception:
                continue
        conn.close()
        return total
    except Exception:
        return 0


def _legacy_db_candidates(target_path: Path) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "data" / "app.db",
        repo_root / ".data" / "app.db",
    ]
    return [p for p in candidates if p.exists() and p.resolve() != target_path.resolve()]


def _copy_legacy_db_if_needed(database_url: str) -> None:
    db_path = _sqlite_path_from_url(database_url)
    if db_path is None:
        return
    if os.getenv("DATABASE_URL"):
        return

    legacy_sources = _legacy_db_candidates(db_path)
    if not legacy_sources:
        return

    source_with_most_rows = max(legacy_sources, key=_sqlite_total_rows)
    source_rows = _sqlite_total_rows(source_with_most_rows)
    target_rows = _sqlite_total_rows(db_path) if db_path.exists() else 0

    should_copy = (not db_path.exists() and source_rows > 0) or (
        db_path.exists() and target_rows == 0 and source_rows > 0
    )
    if not should_copy:
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_with_most_rows, db_path)
    logger.info(
        "Copied legacy SQLite DB from %s to %s (source rows=%s, target rows before copy=%s)",
        source_with_most_rows,
        db_path,
        source_rows,
        target_rows,
    )


_database_url = get_database_url()
_ensure_sqlite_dir(_database_url)
_copy_legacy_db_if_needed(_database_url)

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
