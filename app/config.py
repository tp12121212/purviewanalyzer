import os
from pathlib import Path


def _is_azure_environment() -> bool:
    azure_markers = (
        "CONTAINER_APP_NAME",
        "CONTAINER_APP_REVISION",
        "WEBSITE_INSTANCE_ID",
        "WEBSITE_SITE_NAME",
        "KUBERNETES_SERVICE_HOST",
    )
    return any(os.getenv(marker) for marker in azure_markers)


def _is_writable_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def get_default_db_path() -> Path:
    mnt_root = Path("/mnt")
    if _is_azure_environment() or _is_writable_dir(mnt_root):
        return (mnt_root / "app.db").resolve()
    return (Path.cwd() / "mnt" / "app.db").resolve()


def get_db_path() -> Path:
    configured = os.getenv("DB_PATH")
    path = Path(configured) if configured else get_default_db_path()
    return path.expanduser().resolve()


def get_database_url() -> str:
    configured = os.getenv("DATABASE_URL")
    if configured:
        return configured
    return f"sqlite:///{get_db_path()}"


def get_predefined_recognizers_path() -> Path:
    return Path(os.getenv("PREDEFINED_RECOGNIZERS_PATH", "./predefined_recognizers"))


def auto_import_enabled() -> bool:
    return os.getenv("AUTO_IMPORT_ENTITIES", "0").lower() in {"1", "true", "yes", "on"}
