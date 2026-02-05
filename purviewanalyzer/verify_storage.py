from __future__ import annotations

from app.model_storage import (
    configure_model_caches,
    ensure_directories,
    get_storage_paths,
    verify_mnt_writable,
)


def main() -> int:
    paths = get_storage_paths()
    ensure_directories(paths)
    configure_model_caches(paths)
    verify_mnt_writable(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
