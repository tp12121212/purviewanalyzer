from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import select

from app.config import get_predefined_recognizers_path
from app.db import SessionLocal, init_db
from app.models import Entity, EntityContext, EntityMetadata, EntityPattern

logger = logging.getLogger("presidio-streamlit")

@dataclass
class PatternSpec:
    name: Optional[str]
    regex: Optional[str]
    score: Optional[float]
    order_index: int


@dataclass
class EntitySpec:
    entity_key: str
    name: str
    entity_type: str
    language: Optional[str]
    description: Optional[str]
    recognizer_type: str
    patterns: list[PatternSpec] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    source_hash: str = ""


def _hash_content(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _get_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _eval_constant(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Str):
        return node.s
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.List):
        return [_eval_constant(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_constant(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _eval_constant(key): _eval_constant(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_constant(node.left)
        right = _eval_constant(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def _parse_pattern_call(node: ast.AST) -> Optional[PatternSpec]:
    if not isinstance(node, ast.Call):
        return None
    func_name = _get_name(node.func)
    if func_name != "Pattern":
        return None
    args = [_eval_constant(arg) for arg in node.args]
    kwargs = {kw.arg: _eval_constant(kw.value) for kw in node.keywords if kw.arg}

    name = None
    regex = None
    score = None

    if len(args) >= 1:
        name = args[0]
    if len(args) >= 2:
        regex = args[1]
    if len(args) >= 3:
        score = args[2]

    name = kwargs.get("name", name)
    regex = kwargs.get("regex", regex)
    score = kwargs.get("score", score)

    if name is None and regex is None and score is None:
        return None

    return PatternSpec(
        name=name if isinstance(name, str) else None,
        regex=regex if isinstance(regex, str) else None,
        score=float(score) if isinstance(score, (int, float)) else None,
        order_index=0,
    )


def _extract_class_assignments(class_node: ast.ClassDef) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = item.value
    return assignments


def _extract_init_defaults(class_node: ast.ClassDef) -> dict[str, Any]:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            defaults = [_eval_constant(d) for d in item.args.defaults]
            args = [a.arg for a in item.args.args if a.arg != "self"]
            padded_defaults = [None] * (len(args) - len(defaults)) + defaults
            return dict(zip(args, padded_defaults))
    return {}


def _extract_get_supported_entities(class_node: ast.ClassDef) -> list[str]:
    for item in class_node.body:
        if not isinstance(item, ast.FunctionDef) or item.name != "get_supported_entities":
            continue
        for stmt in item.body:
            if isinstance(stmt, ast.Return):
                value = _eval_constant(stmt.value)
                if isinstance(value, list):
                    return [entry for entry in value if isinstance(entry, str)]
        break
    return []


def _normalize_recognizer_type(base_names: list[str], has_patterns: bool) -> str:
    if has_patterns:
        return "PatternRecognizer"
    priority = ("RemoteRecognizer", "LocalRecognizer", "PatternRecognizer")
    for recognizer_type in priority:
        if recognizer_type in base_names:
            return recognizer_type
    for name in base_names:
        if "Recognizer" in name:
            return name
    return "Recognizer"


def _parse_class_specs(
    class_node: ast.ClassDef,
    source_file: str,
    source_hash: str,
    module_path: str,
) -> list[EntitySpec]:
    base_names = [_get_name(base) for base in class_node.bases]
    if not any("Recognizer" in name for name in base_names):
        return []
    assignments = _extract_class_assignments(class_node)
    init_defaults = _extract_init_defaults(class_node)

    raw_patterns = assignments.get("PATTERNS")
    has_patterns = isinstance(raw_patterns, (ast.List, ast.Tuple))
    recognizer_type = _normalize_recognizer_type(base_names, has_patterns=has_patterns)
    patterns: list[PatternSpec] = []
    if has_patterns:
        for idx, elt in enumerate(raw_patterns.elts):
            spec = _parse_pattern_call(elt)
            if spec:
                spec.order_index = idx
                patterns.append(spec)

    raw_context = assignments.get("CONTEXT")
    context = _eval_constant(raw_context)
    if not isinstance(context, list):
        context = []

    supported_entity = init_defaults.get("supported_entity")
    supported_entities = init_defaults.get("supported_entities")
    supported_language = init_defaults.get("supported_language")

    if supported_entity is None:
        supported_entity = assignments.get("SUPPORTED_ENTITY")
        supported_entity = _eval_constant(supported_entity)
    if supported_entities is None:
        supported_entities = assignments.get("SUPPORTED_ENTITIES")
        supported_entities = _eval_constant(supported_entities)

    if supported_language is None:
        supported_language = assignments.get("SUPPORTED_LANGUAGE")
        supported_language = _eval_constant(supported_language)

    recognizer_name = init_defaults.get("name") or assignments.get("NAME")
    recognizer_name = _eval_constant(recognizer_name) or class_node.name

    description = ast.get_docstring(class_node)

    entity_types: list[str] = []
    if isinstance(supported_entity, str):
        entity_types = [supported_entity]
    elif isinstance(supported_entities, list):
        entity_types = [e for e in supported_entities if isinstance(e, str)]

    if not entity_types:
        entity_types = _extract_get_supported_entities(class_node)
    if not entity_types:
        entity_types = [class_node.name]

    languages: list[Optional[str]] = []
    if isinstance(supported_language, str):
        languages = [supported_language]
    elif isinstance(supported_language, list):
        languages = [lang for lang in supported_language if isinstance(lang, str)]
    else:
        languages = [None]

    specs: list[EntitySpec] = []
    for entity_type in entity_types:
        for language in languages:
            entity_key = f"{source_file}:{class_node.name}:{entity_type}:{language or ''}"
            specs.append(
                EntitySpec(
                    entity_key=entity_key,
                    name=str(recognizer_name),
                    entity_type=entity_type,
                    language=language,
                    description=description,
                    recognizer_type=recognizer_type,
                    patterns=patterns,
                    context=context,
                    metadata={
                        "class_name": class_node.name,
                        "module": module_path,
                        "base_classes": base_names,
                        "parse_method": "ast",
                    },
                    source_file=source_file,
                    source_hash=source_hash,
                )
            )

    return specs


def parse_module_ast(path: Path, root: Path) -> list[EntitySpec]:
    content = path.read_bytes()
    source_hash = _hash_content(content)
    source_file = _relative_path(path, root)
    module_path = source_file.replace("/", ".").replace("\\", ".").rsplit(".py", 1)[0]

    tree = ast.parse(content, filename=str(path))
    specs: list[EntitySpec] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            specs.extend(_parse_class_specs(node, source_file, source_hash, module_path))
    return specs


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Unable to load module {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_module_import(path: Path, root: Path) -> list[EntitySpec]:
    content = path.read_bytes()
    source_hash = _hash_content(content)
    source_file = _relative_path(path, root)
    module_path = source_file.replace("/", ".").replace("\\", ".").rsplit(".py", 1)[0]

    module_name = f"predefined_recognizers_import.{module_path}"
    module = _load_module(path, module_name)

    specs: list[EntitySpec] = []
    for attr in module.__dict__.values():
        if not isinstance(attr, type):
            continue
        if attr.__module__ != module.__name__:
            # Ignore imported classes from presidio_analyzer and third-party modules.
            continue
        if not issubclass(attr, EntityRecognizer) or attr is EntityRecognizer:
            continue
        class_name = attr.__name__
        base_names = [base.__name__ for base in attr.__mro__ if base is not object]
        if not any("Recognizer" in name for name in base_names):
            continue

        init_defaults: dict[str, Any] = {}
        try:
            signature = inspect.signature(attr.__init__)
            for name, param in signature.parameters.items():
                if name == "self" or param.default is inspect._empty:
                    continue
                init_defaults[name] = param.default
        except Exception:
            pass

        description = attr.__doc__
        supported_entity = getattr(attr, "SUPPORTED_ENTITY", init_defaults.get("supported_entity"))
        supported_entities = getattr(attr, "SUPPORTED_ENTITIES", init_defaults.get("supported_entities"))
        context = getattr(attr, "CONTEXT", [])
        patterns_attr = getattr(attr, "PATTERNS", [])
        recognizer_type = _normalize_recognizer_type(
            base_names[1:], has_patterns=bool(patterns_attr)
        )
        supported_language = getattr(attr, "SUPPORTED_LANGUAGE", init_defaults.get("supported_language"))
        recognizer_name = getattr(attr, "NAME", init_defaults.get("name")) or class_name

        entity_types: list[str] = []
        if isinstance(supported_entity, str):
            entity_types = [supported_entity]
        elif isinstance(supported_entities, list):
            entity_types = [e for e in supported_entities if isinstance(e, str)]
        if not entity_types:
            # Skip classes where entity mapping could not be resolved.
            continue

        if isinstance(supported_language, str):
            languages = [supported_language]
        elif isinstance(supported_language, list):
            languages = [lang for lang in supported_language if isinstance(lang, str)]
        else:
            languages = [None]

        patterns: list[PatternSpec] = []
        for idx, pattern in enumerate(patterns_attr or []):
            name = getattr(pattern, "name", None)
            regex = getattr(pattern, "regex", None)
            score = getattr(pattern, "score", None)
            patterns.append(
                PatternSpec(
                    name=name,
                    regex=regex,
                    score=score,
                    order_index=idx,
                )
            )

        for entity_type in entity_types:
            for language in languages:
                entity_key = f"{source_file}:{class_name}:{entity_type}:{language or ''}"
                specs.append(
                    EntitySpec(
                        entity_key=entity_key,
                        name=str(recognizer_name),
                        entity_type=entity_type,
                        language=language,
                        description=description,
                        recognizer_type=recognizer_type,
                        patterns=patterns,
                        context=context or [],
                        metadata={
                            "class_name": class_name,
                            "module": module_path,
                            "base_classes": base_names,
                            "parse_method": "import",
                        },
                        source_file=source_file,
                        source_hash=source_hash,
                    )
                )

    return specs


def discover_recognizer_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*.py")
        if path.is_file() and path.name != "__init__.py"
    ]


def load_entity_specs(
    root: Path, import_strategy: str = "auto"
) -> list[EntitySpec]:
    specs: list[EntitySpec] = []
    allow_import_exec = os.getenv("ALLOW_UNSAFE_RECOGNIZER_IMPORTS", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for path in discover_recognizer_files(root):
        try:
            if import_strategy == "import" or (
                import_strategy == "auto" and allow_import_exec
            ):
                specs.extend(parse_module_import(path, root))
            else:
                specs.extend(parse_module_ast(path, root))
        except Exception as exc:
            logger.warning("Failed to import recognizer module %s: %s", path, exc)
            specs.extend(parse_module_ast(path, root))
    return specs


def upsert_entities(
    specs: list[EntitySpec], discovered_source_files: Optional[list[str]] = None
) -> int:
    init_db()
    inserted = 0
    spec_keys = {spec.entity_key for spec in specs}
    spec_source_files = sorted({spec.source_file for spec in specs if spec.source_file})
    discovered_source_files = sorted(set(discovered_source_files or spec_source_files))
    with SessionLocal() as session:
        # Defensive schema creation for test environments that reload app.db/app.models.
        Entity.__table__.create(bind=session.bind, checkfirst=True)
        EntityPattern.__table__.create(bind=session.bind, checkfirst=True)
        EntityContext.__table__.create(bind=session.bind, checkfirst=True)
        EntityMetadata.__table__.create(bind=session.bind, checkfirst=True)

        if discovered_source_files:
            # Remove entries for recognizer files that no longer exist.
            session.query(Entity).filter(
                Entity.source == "predefined_recognizers",
                ~Entity.source_file.in_(discovered_source_files),
            ).delete(synchronize_session=False)

            # Remove stale rows for existing files when entity keys no longer match.
            stale_query = session.query(Entity).filter(
                Entity.source == "predefined_recognizers",
                Entity.source_file.in_(discovered_source_files),
            )
            if spec_keys:
                stale_query = stale_query.filter(~Entity.entity_key.in_(spec_keys))
            stale_query.delete(synchronize_session=False)

        for spec in specs:
            entity = session.execute(
                select(Entity).where(Entity.entity_key == spec.entity_key)
            ).scalar_one_or_none()

            if entity is None:
                entity = Entity(
                    entity_key=spec.entity_key,
                    name=spec.name,
                    entity_type=spec.entity_type,
                    language=spec.language,
                    description=spec.description,
                    recognizer_type=spec.recognizer_type,
                    enabled=True,
                    source="predefined_recognizers",
                    source_file=spec.source_file,
                    source_hash=spec.source_hash,
                )
                session.add(entity)
                session.flush()
                inserted += 1
            else:
                entity.name = spec.name
                entity.entity_type = spec.entity_type
                entity.language = spec.language
                entity.description = spec.description
                entity.recognizer_type = spec.recognizer_type
                entity.source = "predefined_recognizers"
                entity.source_file = spec.source_file
                entity.source_hash = spec.source_hash

                session.query(EntityPattern).filter(
                    EntityPattern.entity_id == entity.id
                ).delete()
                session.query(EntityContext).filter(
                    EntityContext.entity_id == entity.id
                ).delete()
                session.query(EntityMetadata).filter(
                    EntityMetadata.entity_id == entity.id
                ).delete()

            for pattern in spec.patterns:
                session.add(
                    EntityPattern(
                        entity_id=entity.id,
                        pattern_name=pattern.name,
                        regex=pattern.regex,
                        score=pattern.score,
                        order_index=pattern.order_index,
                    )
                )

            for context_item in spec.context:
                if not isinstance(context_item, str):
                    continue
                session.add(
                    EntityContext(entity_id=entity.id, context=context_item)
                )

            for key, value in spec.metadata.items():
                session.add(
                    EntityMetadata(
                        entity_id=entity.id,
                        key=key,
                        value_json=json.dumps(value),
                    )
                )

        session.commit()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Import predefined recognizers into DB")
    parser.add_argument(
        "--path",
        default=str(get_predefined_recognizers_path()),
        help="Path to predefined recognizers",
    )
    parser.add_argument(
        "--import-strategy",
        choices=["auto", "import", "ast"],
        default="auto",
        help="Import strategy for parsing recognizer modules",
    )
    args = parser.parse_args()

    root = Path(args.path)
    discovered_paths = [
        _relative_path(path, root) for path in discover_recognizer_files(root)
    ]
    specs = load_entity_specs(root, import_strategy=args.import_strategy)
    inserted = upsert_entities(specs, discovered_source_files=discovered_paths)
    print(f"Imported {len(specs)} entities ({inserted} new).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
