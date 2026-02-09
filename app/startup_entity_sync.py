from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

from app.config import get_db_path, get_predefined_recognizers_path
from app.import_entities import discover_recognizer_files, load_entity_specs, upsert_entities

logger = logging.getLogger("presidio-streamlit")


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


@contextmanager
def _exclusive_lock(lock_path: Path, timeout_seconds: int):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            import fcntl
        except Exception:
            # Non-POSIX fallback: proceed without lock.
            yield
            return

        deadline = time.time() + max(timeout_seconds, 1)
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for entity sync lock: {lock_path}"
                    )
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def sync_entities(import_strategy: str = "ast") -> int:
    recognizers_root = get_predefined_recognizers_path()
    discovered_files = [
        str(path.relative_to(recognizers_root))
        for path in discover_recognizer_files(recognizers_root)
    ]
    specs = load_entity_specs(recognizers_root, import_strategy=import_strategy)
    inserted = upsert_entities(specs, discovered_source_files=discovered_files)
    logger.info("Entity sync complete: scanned=%s inserted=%s", len(specs), inserted)
    return len(specs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize entities DB from predefined recognizer source files"
    )
    parser.add_argument(
        "--import-strategy",
        choices=["auto", "import", "ast"],
        default=os.getenv("ENTITY_SYNC_IMPORT_STRATEGY", "ast"),
    )
    args = parser.parse_args()

    if not _env_enabled("STARTUP_ENTITY_SYNC", "1"):
        logger.info("Startup entity sync disabled via STARTUP_ENTITY_SYNC.")
        return 0

    db_path = get_db_path()
    lock_path = db_path.parent / ".entity_sync.lock"
    timeout_seconds = int(os.getenv("ENTITY_SYNC_LOCK_TIMEOUT_SECONDS", "60"))
    strict_mode = _env_enabled("ENTITY_SYNC_STRICT", "1")

    try:
        with _exclusive_lock(lock_path, timeout_seconds=timeout_seconds):
            sync_entities(import_strategy=args.import_strategy)
        return 0
    except Exception as exc:
        logger.exception("Startup entity sync failed: %s", exc)
        if strict_mode:
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
