from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Entity, EntityContext, EntityMetadata, EntityPattern


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
    items = session.execute(
        query.order_by(Entity.name.asc()).offset(offset).limit(page_size)
    ).scalars()

    pattern_counts = dict(
        session.execute(
            select(EntityPattern.entity_id, func.count()).group_by(EntityPattern.entity_id)
        ).all()
    )

    results = []
    for item in items:
        results.append(
            {
                "id": item.id,
                "name": item.name,
                "entity_type": item.entity_type,
                "language": item.language,
                "recognizer_type": item.recognizer_type,
                "enabled": item.enabled,
                "updated_at": item.updated_at,
                "patterns_count": pattern_counts.get(item.id, 0),
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
        "metadata": {key: value for key, value in metadata_items},
    }
