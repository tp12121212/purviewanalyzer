from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Entity, EntityContext, EntityMetadata, EntityPattern, Recognizer


def _decode_metadata_value(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str)]
    return []


def list_entities(
    session: Session,
    search: Optional[str] = None,
    language: Optional[str] = None,
    entity_type: Optional[str] = None,
    enabled: Optional[bool] = None,
    recognizer_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    query = select(Entity)

    if search:
        like = f"%{search}%"
        query = query.where(
            (Entity.name.ilike(like))
            | (Entity.entity_type.ilike(like))
            | (Entity.source_file.ilike(like))
        )
    if language:
        query = query.where(Entity.language == language)
    if entity_type:
        query = query.where(Entity.entity_type == entity_type)
    if enabled is not None:
        query = query.where(Entity.enabled == enabled)
    if recognizer_type:
        query = query.where(Entity.recognizer_type == recognizer_type)

    total = session.execute(select(func.count()).select_from(query.subquery())).scalar()

    offset = max(page - 1, 0) * page_size
    items = list(
        session.execute(
            query.order_by(Entity.name.asc()).offset(offset).limit(page_size)
        ).scalars()
    )
    item_ids = [item.id for item in items]

    if item_ids:
        pattern_counts = dict(
            session.execute(
                select(EntityPattern.entity_id, func.count())
                .where(EntityPattern.entity_id.in_(item_ids))
                .group_by(EntityPattern.entity_id)
            ).all()
        )
        context_counts = dict(
            session.execute(
                select(EntityContext.entity_id, func.count())
                .where(EntityContext.entity_id.in_(item_ids))
                .group_by(EntityContext.entity_id)
            ).all()
        )
        metadata_counts = dict(
            session.execute(
                select(EntityMetadata.entity_id, func.count())
                .where(EntityMetadata.entity_id.in_(item_ids))
                .group_by(EntityMetadata.entity_id)
            ).all()
        )
    else:
        pattern_counts = {}
        context_counts = {}
        metadata_counts = {}

    results = []
    for item in items:
        patterns_count = pattern_counts.get(item.id, 0)
        context_count = context_counts.get(item.id, 0)
        metadata_count = metadata_counts.get(item.id, 0)
        results.append(
            {
                "id": item.id,
                "name": item.name,
                "entity_type": item.entity_type,
                "language": item.language,
                "recognizer_type": item.recognizer_type,
                "enabled": item.enabled,
                "updated_at": item.updated_at,
                "patterns_count": patterns_count,
                "context_count": context_count,
                "metadata_count": metadata_count,
            }
        )

    return {"total": total or 0, "items": results, "page": page, "page_size": page_size}


def get_entity_detail(session: Session, entity_id: int) -> Optional[dict]:
    entity = session.get(Entity, entity_id)
    if not entity:
        return None

    patterns = session.execute(
        select(EntityPattern)
        .where(EntityPattern.entity_id == entity_id)
        .order_by(EntityPattern.order_index.asc())
    ).scalars()

    contexts = session.execute(
        select(EntityContext.context)
        .where(EntityContext.entity_id == entity_id)
        .order_by(EntityContext.context.asc())
    ).scalars()

    metadata_items = session.execute(
        select(EntityMetadata.key, EntityMetadata.value_json).where(
            EntityMetadata.entity_id == entity_id
        )
    ).all()
    metadata = {key: _decode_metadata_value(value) for key, value in metadata_items}

    class_name = metadata.get("class_name")
    recognizer_query = select(Recognizer).where(
        Recognizer.module_path == entity.source_file,
        Recognizer.entity_type == entity.entity_type,
        func.coalesce(Recognizer.language, "") == (entity.language or ""),
    )
    if isinstance(class_name, str) and class_name:
        recognizer_query = recognizer_query.where(Recognizer.class_name == class_name)
    match_words: list[str] = []
    recognizer = session.execute(recognizer_query).scalar_one_or_none()
    if recognizer:
        match_words = _parse_json_list(recognizer.deny_list_json)

    return {
        "id": entity.id,
        "name": entity.name,
        "entity_type": entity.entity_type,
        "language": entity.language,
        "description": entity.description,
        "recognizer_type": entity.recognizer_type,
        "enabled": entity.enabled,
        "source": entity.source,
        "source_file": entity.source_file,
        "source_hash": entity.source_hash,
        "created_at": entity.created_at,
        "updated_at": entity.updated_at,
        "patterns": [
            {
                "pattern_name": pattern.pattern_name,
                "regex": pattern.regex,
                "score": pattern.score,
            }
            for pattern in patterns
        ],
        "context": list(contexts),
        "match_words": match_words,
        "metadata": metadata,
    }
