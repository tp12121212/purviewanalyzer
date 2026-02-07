from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from presidio_analyzer import Pattern, PatternRecognizer

from app.recognizers_service import RecognizerValidationError, validate_patterns


@dataclass(frozen=True)
class AdHocRecognizerSpec:
    name: str
    entity_type: str
    language: str | None
    base_score: float | None
    patterns: list[dict]
    context: list[str]
    allow_list: list[str] | None
    deny_list: list[str] | None


def _ensure_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def parse_ad_hoc_json(payload: str) -> list[AdHocRecognizerSpec]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RecognizerValidationError(f"Invalid JSON: {exc}") from exc

    recognizers = data.get("recognizers") if isinstance(data, dict) else data
    if not isinstance(recognizers, list):
        raise RecognizerValidationError("Expected a list of recognizers.")

    specs: list[AdHocRecognizerSpec] = []
    for idx, item in enumerate(recognizers):
        if not isinstance(item, dict):
            raise RecognizerValidationError(
                f"Recognizer {idx + 1} must be an object."
            )
        name = str(item.get("name", "")).strip()
        entity_types = _ensure_list(item.get("supported_entities") or item.get("entity_type"))
        if not entity_types:
            raise RecognizerValidationError(
                f"Recognizer {idx + 1} is missing supported_entities."
            )
        language = item.get("supported_language") or item.get("language")
        base_score = item.get("base_score")
        patterns = item.get("patterns") or []
        context = item.get("context") or []
        allow_list = item.get("allow_list")
        deny_list = item.get("deny_list")

        if not isinstance(patterns, list):
            raise RecognizerValidationError(
                f"Recognizer {idx + 1} patterns must be a list."
            )

        validate_patterns(patterns, base_score, allow_empty=bool(deny_list))

        for entity in entity_types:
            specs.append(
                AdHocRecognizerSpec(
                    name=name or f"AdHoc{entity}",
                    entity_type=str(entity),
                    language=str(language) if language else None,
                    base_score=base_score,
                    patterns=patterns,
                    context=[str(c) for c in context if isinstance(c, str)],
                    allow_list=allow_list if isinstance(allow_list, list) else None,
                    deny_list=deny_list if isinstance(deny_list, list) else None,
                )
            )

    return specs


def build_ad_hoc_recognizers(specs: list[AdHocRecognizerSpec]) -> list[PatternRecognizer]:
    recognizers: list[PatternRecognizer] = []
    for spec in specs:
        patterns = []
        for pattern in spec.patterns:
            score = pattern.get("score", spec.base_score)
            patterns.append(
                Pattern(
                    name=pattern.get("name") or "Pattern",
                    regex=pattern.get("regex") or "",
                    score=score if score is not None else 0.5,
                )
            )
        recognizers.append(
            PatternRecognizer(
                name=spec.name,
                supported_entity=spec.entity_type,
                supported_language=spec.language,
                patterns=patterns,
                context=spec.context,
                deny_list=spec.deny_list,
            )
        )
    return recognizers
