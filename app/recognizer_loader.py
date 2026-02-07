from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import Iterable

from app.db import SessionLocal, init_db
from app.recognizer_codegen import resolve_storage_path
from app.recognizers_service import iter_enabled_recognizers

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


def _migrate_allow_list_kwarg(path: Path) -> bool:
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if "allow_list=" not in content:
        return False
    updated = re.sub(r"^\s*allow_list=.*\n", "", content, flags=re.MULTILINE)
    if updated == content:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


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
            # Avoid instantiating imported base classes such as PatternRecognizer.
            if attr.__module__ != module.__name__:
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


def load_persistent_recognizers(registry) -> None:
    init_db()
    with SessionLocal() as session:
        recognizers = list(iter_enabled_recognizers(session))

    if not recognizers:
        return

    already_loaded = {rec.__class__.__name__ for rec in registry.recognizers}
    for recognizer in recognizers:
        if recognizer.class_name in already_loaded:
            continue
        root = Path(recognizer.storage_root)
        try:
            path = resolve_storage_path(root, recognizer.storage_subpath, Path(recognizer.module_path).name)
        except ValueError as exc:
            logger.warning(
                "Skipping recognizer %s due to unsafe path: %s",
                recognizer.name,
                exc,
            )
            continue

        module_name = f"custom_recognizers.{recognizer.class_name}"
        try:
            module = _load_module(path, module_name)
        except Exception as exc:
            logger.warning("Failed to import custom recognizer module %s: %s", path, exc)
            continue

        recognizer_cls = module.__dict__.get(recognizer.class_name)
        if not isinstance(recognizer_cls, type):
            logger.warning("Missing class %s in module %s", recognizer.class_name, path)
            continue

        if not issubclass(recognizer_cls, EntityRecognizer):
            logger.warning("Class %s is not an EntityRecognizer", recognizer.class_name)
            continue

        try:
            instance = recognizer_cls()
        except Exception as exc:
            if "allow_list" in str(exc) and _migrate_allow_list_kwarg(path):
                try:
                    module = _load_module(path, f"{module_name}.migrated")
                    recognizer_cls = module.__dict__.get(recognizer.class_name)
                    if isinstance(recognizer_cls, type) and issubclass(recognizer_cls, EntityRecognizer):
                        instance = recognizer_cls()
                        registry.add_recognizer(instance)
                        already_loaded.add(recognizer.class_name)
                        logger.info("Registered persistent recognizer %s", recognizer.class_name)
                        continue
                except Exception:
                    pass
            logger.warning(
                "Failed to instantiate recognizer %s from %s: %s",
                recognizer.class_name,
                path,
                exc,
            )
            continue

        registry.add_recognizer(instance)
        already_loaded.add(recognizer.class_name)
        logger.info("Registered persistent recognizer %s", recognizer.class_name)
