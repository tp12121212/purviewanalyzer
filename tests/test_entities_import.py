import importlib
import os
from pathlib import Path


def _setup_env(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    recognizers_path = repo_root / "predefined_recognizers"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}/test.db"
    os.environ["PREDEFINED_RECOGNIZERS_PATH"] = str(recognizers_path)
    return recognizers_path


def test_import_upsert(tmp_path):
    recognizers_path = _setup_env(tmp_path)

    import app.db as db
    importlib.reload(db)
    import app.import_entities as importer
    importlib.reload(importer)
    import app.models as models
    importlib.reload(models)

    specs = importer.load_entity_specs(recognizers_path, import_strategy="ast")
    assert specs

    inserted_first = importer.upsert_entities(specs)
    inserted_second = importer.upsert_entities(specs)

    assert inserted_first > 0
    assert inserted_second == 0

    with db.SessionLocal() as session:
        total = session.query(models.Entity).count()

    assert total == len({spec.entity_key for spec in specs})


def test_api_entities(tmp_path):
    recognizers_path = _setup_env(tmp_path)

    import app.db as db
    importlib.reload(db)
    import app.import_entities as importer
    importlib.reload(importer)
    importer.upsert_entities(importer.load_entity_specs(recognizers_path, "ast"))

    import app.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient

    client = TestClient(api.app)
    response = client.get("/api/entities")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert data["items"]

    entity_id = data["items"][0]["id"]
    detail = client.get(f"/api/entities/{entity_id}")
    assert detail.status_code == 200
    detail_data = detail.json()
    assert detail_data["id"] == entity_id
    assert "patterns" in detail_data
