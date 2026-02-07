import os
from pathlib import Path


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./data/app.db")


def get_predefined_recognizers_path() -> Path:
    return Path(os.getenv("PREDEFINED_RECOGNIZERS_PATH", "./predefined_recognizers"))


def get_custom_recognizers_root() -> Path:
    return Path(
        os.getenv(
            "CUSTOM_RECOGNIZERS_ROOT",
            os.getenv("PREDEFINED_RECOGNIZERS_PATH", "./predefined_recognizers"),
        )
    )


def auto_import_enabled() -> bool:
    return os.getenv("AUTO_IMPORT_ENTITIES", "0").lower() in {"1", "true", "yes", "on"}
