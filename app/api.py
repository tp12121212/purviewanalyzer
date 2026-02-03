from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.entities_service import get_entity_detail, list_entities

app = FastAPI(title="Presidio Entities API")


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/api/entities")
def api_list_entities(
    search: str | None = None,
    language: str | None = None,
    entity_type: str | None = None,
    enabled: bool | None = None,
    recognizer_type: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    session: Session = Depends(get_session),
):
    return list_entities(
        session,
        search=search,
        language=language,
        entity_type=entity_type,
        enabled=enabled,
        recognizer_type=recognizer_type,
        page=page,
        page_size=page_size,
    )


@app.get("/api/entities/{entity_id}")
def api_get_entity(entity_id: int, session: Session = Depends(get_session)):
    entity = get_entity_detail(session, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity
