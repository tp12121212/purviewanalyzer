from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
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


def _copy_legacy_db_if_needed(database_url: str) -> None:
    db_path = _sqlite_path_from_url(database_url)
    if db_path is None:
        return
    if os.getenv("DATABASE_URL"):
        return
    if db_path.exists():
        return

    legacy_db_path = Path(__file__).resolve().parents[1] / "data" / "app.db"
    if db_path.resolve() == legacy_db_path.resolve():
        return
    if not legacy_db_path.exists():
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_db_path, db_path)
    logger.info("Copied legacy SQLite DB from %s to %s", legacy_db_path, db_path)


_database_url = get_database_url()
_ensure_sqlite_dir(_database_url)
_copy_legacy_db_if_needed(_database_url)

engine = create_engine(
    _database_url,
    connect_args={"check_same_thread": False} if _database_url.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_session() -> Generator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
