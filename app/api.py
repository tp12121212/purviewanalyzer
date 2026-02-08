from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.model_storage import init_model_storage
from app.entities_service import get_entity_detail, list_entities

app = FastAPI(title="Presidio Entities API")


def _require_api_token(request: Request) -> None:
    token = os.getenv("API_AUTH_TOKEN")
    if not token:
        return
    auth_header = request.headers.get("Authorization", "")
    header_token = ""
    if auth_header.lower().startswith("bearer "):
        header_token = auth_header.split(" ", 1)[1].strip()
    if not header_token:
        header_token = request.headers.get("X-API-KEY", "").strip()
    if header_token != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@app.on_event("startup")
def _startup() -> None:
    init_model_storage()
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
    _auth: None = Depends(_require_api_token),
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
def api_get_entity(
    entity_id: int,
    session: Session = Depends(get_session),
    _auth: None = Depends(_require_api_token),
):
    entity = get_entity_detail(session, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return entity
