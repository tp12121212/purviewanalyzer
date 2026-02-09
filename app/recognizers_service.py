from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_predefined_recognizers_path
from app.models import (
    Entity,
    EntityContext,
    EntityMetadata,
    EntityPattern,
    Recognizer,
    RecognizerContext,
    RecognizerPattern,
)
from app.recognizer_codegen import (
    RecognizerCodegenSpec,
    derive_class_name,
    derive_module_filename,
    generate_recognizer_module,
    hash_code,
    resolve_storage_path,
    update_init_exports,
    write_recognizer_module,
)


class RecognizerValidationError(ValueError):
    pass


@dataclass(frozen=True)
class RecognizerInput:
    name: str
    entity_type: str
    class_name: str | None
    module_filename: str | None
    language: str | None
    description: str | None
    enabled: bool
    base_score: float | None
    patterns: list[dict]
    context: list[str]
    allow_list: list[str] | None
    deny_list: list[str] | None
    storage_subpath: str | None
    version: str | None


def _json_or_none(value: list[str] | None) -> str | None:
    if not value:
        return None
    return json.dumps(value)


def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def _looks_unsafe_regex(pattern: str) -> bool:
    # Heuristic to avoid obvious catastrophic backtracking patterns like (.+)+ or (.*)*
    nested_quantifiers = re.search(r"\((?:[^()]*[+*][^()]*)\)[+*]", pattern)
    return bool(nested_quantifiers)


def validate_patterns(
    patterns: list[dict], base_score: float | None, allow_empty: bool = False
) -> None:
    if not patterns:
        if allow_empty:
            return
        raise RecognizerValidationError(
            "Provide at least one regex pattern, or add one or more match words."
        )

    for idx, pattern in enumerate(patterns):
        regex = pattern.get("regex")
        if not regex:
            raise RecognizerValidationError(
                f"Pattern {idx + 1} is missing a regex value. Remove this row or provide a regex."
            )
        try:
            re.compile(regex)
        except re.error as exc:
            raise RecognizerValidationError(
                f"Pattern {idx + 1} failed to compile: {exc}"
            ) from exc
        if _looks_unsafe_regex(regex):
            raise RecognizerValidationError(
                f"Pattern {idx + 1} appears unsafe (nested quantifiers)."
            )
        score = pattern.get("score", base_score)
        if score is not None and (score < 0 or score > 1):
            raise RecognizerValidationError(
                f"Pattern {idx + 1} score must be between 0 and 1."
            )


def validate_recognizer_input(data: RecognizerInput) -> None:
    if not data.name.strip():
        raise RecognizerValidationError("Recognizer name is required.")
    if not data.entity_type.strip():
        raise RecognizerValidationError("Supported entity key is required.")
    if data.base_score is not None and (data.base_score < 0 or data.base_score > 1):
        raise RecognizerValidationError("Base score must be between 0 and 1.")
    validate_patterns(
        data.patterns, data.base_score, allow_empty=bool(data.deny_list)
    )
    if data.class_name is not None and not data.class_name.strip():
        raise RecognizerValidationError("Recognizer class cannot be empty.")
    if data.module_filename is not None and not data.module_filename.strip():
        raise RecognizerValidationError("File name cannot be empty.")


def _sanitize_class_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "", value.strip())
    if not cleaned:
        raise RecognizerValidationError("Recognizer class contains no valid characters.")
    if not cleaned[0].isalpha():
        cleaned = f"C{cleaned}"
    if not cleaned.endswith("Recognizer"):
        cleaned += "Recognizer"
    return cleaned


def _sanitize_module_filename(value: str, class_name: str) -> str:
    filename = value.strip()
    if not filename:
        raise RecognizerValidationError("File name is required.")
    if "/" in filename or "\\" in filename:
        raise RecognizerValidationError("File name must not contain path separators.")
    if not filename.endswith(".py"):
        filename += ".py"
    stem = Path(filename).stem
    if not stem:
        raise RecognizerValidationError("File name stem is required.")
    safe_stem = re.sub(r"[^A-Za-z0-9_]", "_", stem).strip("_").lower()
    if not safe_stem:
        safe_stem = derive_module_filename(class_name).rsplit(".py", 1)[0]
    return f"{safe_stem}.py"


def _compute_module_path(subpath: str | None, filename: str) -> str:
    cleaned = (subpath or "generic").strip().strip("/")
    if not cleaned:
        cleaned = "generic"
    if cleaned:
        return f"{cleaned}/{filename}"
    return filename


def create_or_update_recognizer(
    session: Session, data: RecognizerInput, recognizer_id: int | None = None
) -> Recognizer:
    validate_recognizer_input(data)

    storage_root = get_predefined_recognizers_path()
    class_name = (
        _sanitize_class_name(data.class_name)
        if data.class_name
        else derive_class_name(data.name, data.entity_type)
    )
    filename = (
        _sanitize_module_filename(data.module_filename, class_name)
        if data.module_filename
        else derive_module_filename(class_name)
    )
    effective_subpath = (data.storage_subpath or "generic").strip().strip("/")
    if not effective_subpath:
        effective_subpath = "generic"

    module_path = _compute_module_path(effective_subpath, filename)
    try:
        file_path = resolve_storage_path(storage_root, effective_subpath, filename)
    except ValueError as exc:
        raise RecognizerValidationError(str(exc)) from exc

    if recognizer_id is None and file_path.exists():
        raise RecognizerValidationError(
            "Recognizer module already exists. Choose a different name or subfolder."
        )

    spec = RecognizerCodegenSpec(
        name=data.name,
        class_name=class_name,
        entity_type=data.entity_type,
        language=data.language,
        description=data.description,
        base_score=data.base_score,
        patterns=data.patterns,
        context=data.context,
        allow_list=data.allow_list,
        deny_list=data.deny_list,
    )
    code = generate_recognizer_module(spec)
    write_recognizer_module(file_path, code)
    update_init_exports(storage_root, effective_subpath, filename, class_name)
    source_hash = hash_code(code)

    recognizer = None
    if recognizer_id is not None:
        recognizer = session.get(Recognizer, recognizer_id)

    if recognizer is None:
        recognizer = Recognizer(
            name=data.name,
            entity_type=data.entity_type,
            language=data.language,
            description=data.description,
            enabled=data.enabled,
            base_score=data.base_score,
            allow_list_json=_json_or_none(data.allow_list),
            deny_list_json=_json_or_none(data.deny_list),
            storage_root=str(storage_root),
            storage_subpath=effective_subpath,
            module_path=module_path,
            class_name=class_name,
            version=data.version,
        )
        session.add(recognizer)
        session.flush()
    else:
        recognizer.name = data.name
        recognizer.entity_type = data.entity_type
        recognizer.language = data.language
        recognizer.description = data.description
        recognizer.enabled = data.enabled
        recognizer.base_score = data.base_score
        recognizer.allow_list_json = _json_or_none(data.allow_list)
        recognizer.deny_list_json = _json_or_none(data.deny_list)
        recognizer.storage_root = str(storage_root)
        recognizer.storage_subpath = effective_subpath
        recognizer.module_path = module_path
        recognizer.class_name = class_name
        recognizer.version = data.version

        session.query(RecognizerPattern).filter(
            RecognizerPattern.recognizer_id == recognizer.id
        ).delete()
        session.query(RecognizerContext).filter(
            RecognizerContext.recognizer_id == recognizer.id
        ).delete()

    for idx, pattern in enumerate(data.patterns):
        session.add(
            RecognizerPattern(
                recognizer_id=recognizer.id,
                pattern_name=pattern.get("name"),
                regex=pattern.get("regex"),
                score=pattern.get("score", data.base_score),
                order_index=idx,
            )
        )

    for context_item in data.context:
        if not isinstance(context_item, str):
            continue
        session.add(
            RecognizerContext(recognizer_id=recognizer.id, context=context_item)
        )

    _upsert_entity_from_recognizer(
        session=session,
        recognizer=recognizer,
        patterns=data.patterns,
        context=data.context,
        source_hash=source_hash,
    )

    session.commit()
    return recognizer


def _upsert_entity_from_recognizer(
    session: Session,
    recognizer: Recognizer,
    patterns: list[dict],
    context: list[str],
    source_hash: str,
) -> None:
    entity_key = (
        f"{recognizer.module_path}:{recognizer.class_name}:"
        f"{recognizer.entity_type}:{recognizer.language or ''}"
    )
    entity = session.execute(
        select(Entity).where(Entity.entity_key == entity_key)
    ).scalar_one_or_none()

    if entity is None:
        entity = Entity(
            entity_key=entity_key,
            name=recognizer.name,
            entity_type=recognizer.entity_type,
            language=recognizer.language,
            description=recognizer.description,
            recognizer_type="PatternRecognizer",
            enabled=recognizer.enabled,
            source="predefined_recognizers",
            source_file=recognizer.module_path,
            source_hash=source_hash,
        )
        session.add(entity)
        session.flush()
    else:
        entity.name = recognizer.name
        entity.entity_type = recognizer.entity_type
        entity.language = recognizer.language
        entity.description = recognizer.description
        entity.recognizer_type = "PatternRecognizer"
        entity.enabled = recognizer.enabled
        entity.source = "predefined_recognizers"
        entity.source_file = recognizer.module_path
        entity.source_hash = source_hash

        session.query(EntityPattern).filter(EntityPattern.entity_id == entity.id).delete()
        session.query(EntityContext).filter(EntityContext.entity_id == entity.id).delete()
        session.query(EntityMetadata).filter(EntityMetadata.entity_id == entity.id).delete()

    for idx, pattern in enumerate(patterns):
        session.add(
            EntityPattern(
                entity_id=entity.id,
                pattern_name=pattern.get("name"),
                regex=pattern.get("regex"),
                score=pattern.get("score"),
                order_index=idx,
            )
        )

    for context_item in context:
        if not isinstance(context_item, str):
            continue
        session.add(EntityContext(entity_id=entity.id, context=context_item))

    metadata = {
        "class_name": recognizer.class_name,
        "module": recognizer.module_path.replace("/", ".").rsplit(".py", 1)[0],
        "base_score": recognizer.base_score,
    }
    for key, value in metadata.items():
        session.add(
            EntityMetadata(
                entity_id=entity.id, key=key, value_json=json.dumps(value)
            )
        )


def delete_recognizer(session: Session, recognizer_id: int) -> None:
    recognizer = session.get(Recognizer, recognizer_id)
    if recognizer is None:
        raise RecognizerValidationError("Recognizer not found.")
    entity_key = (
        f"{recognizer.module_path}:{recognizer.class_name}:"
        f"{recognizer.entity_type}:{recognizer.language or ''}"
    )
    session.query(Entity).filter(Entity.entity_key == entity_key).delete()
    session.delete(recognizer)
    session.commit()


def list_recognizers(session: Session) -> list[Recognizer]:
    return list(session.execute(select(Recognizer).order_by(Recognizer.name.asc())).scalars())


def get_recognizer_detail(session: Session, recognizer_id: int) -> Recognizer | None:
    return session.get(Recognizer, recognizer_id)


def iter_enabled_recognizers(session: Session) -> Iterable[Recognizer]:
    return session.execute(
        select(Recognizer).where(Recognizer.enabled == True)  # noqa: E712
    ).scalars()


def unpack_allow_list(recognizer: Recognizer) -> list[str]:
    return _parse_json_list(recognizer.allow_list_json)


def unpack_deny_list(recognizer: Recognizer) -> list[str]:
    return _parse_json_list(recognizer.deny_list_json)
