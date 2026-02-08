from pathlib import Path

from app import config


def test_local_default_db_path(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "_is_azure_environment", lambda: False)
    monkeypatch.setattr(config, "_is_writable_dir", lambda _path: False)

    expected = (Path.cwd() / "mnt" / "app.db").resolve()
    assert config.get_db_path() == expected


def test_mnt_default_db_path(monkeypatch):
    monkeypatch.delenv("DB_PATH", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(config, "_is_azure_environment", lambda: False)
    monkeypatch.setattr(config, "_is_writable_dir", lambda _path: True)

    assert config.get_db_path() == Path("/mnt/app.db")


def test_database_url_prefers_database_url_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "sqlite:////tmp/custom.db")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "ignored.db"))

    assert config.get_database_url() == "sqlite:////tmp/custom.db"
