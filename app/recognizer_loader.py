from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable

from presidio_analyzer import EntityRecognizer

logger = logging.getLogger("presidio-streamlit")


def _discover_country_specific_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and path.name != "__init__.py"
    )


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Unable to load module {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_country_specific_recognizers(registry) -> None:
    root = Path(__file__).resolve().parents[1] / "predefined_recognizers" / "country_specific"
    logger.info("Discovering country-specific recognizers under %s", root)
    for path in _discover_country_specific_paths(root):
        module_path = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        module_name = f"local_country_specific.{module_path}"
        logger.debug("Discovered recognizer module %s", path)
        try:
            module = _load_module(path, module_name)
        except Exception as exc:
            logger.warning("Failed to import recognizer module %s: %s", path, exc)
            continue

        for attr in module.__dict__.values():
            if not isinstance(attr, type):
                continue
            if not issubclass(attr, EntityRecognizer) or attr is EntityRecognizer:
                continue
            try:
                recognizer = attr()
            except Exception as exc:
                logger.warning(
                    "Failed to instantiate recognizer %s from %s: %s",
                    attr.__name__,
                    path,
                    exc,
                )
                continue
            registry.add_recognizer(recognizer)
            logger.info(
                "Registered recognizer %s from %s", recognizer.__class__.__name__, path
            )
