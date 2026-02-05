from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger("purview-model-storage")

PERSIST_ROOT_ENV = "PURVIEW_PERSIST_ROOT"
VERIFY_MNT_WRITABLE_ENV = "VERIFY_MNT_WRITABLE"
CLEAN_LEGACY_ENV = "PURVIEW_CLEAN_LEGACY_MODEL_DIRS"

DEFAULT_PERSIST_ROOT = Path("/mnt")


@dataclass(frozen=True)
class ModelStoragePaths:
    persist_root: Path
    models_dir: Path
    spacy_dir: Path
    hf_dir: Path
    flair_dir: Path
    stanza_dir: Path
    xdg_cache_dir: Path
    torch_dir: Path

    @property
    def is_ephemeral(self) -> bool:
        return self.persist_root.name.startswith("purview-persist-") and self.persist_root.is_dir()


_storage_paths: ModelStoragePaths | None = None
_initialized = False


def _resolve_persist_root() -> Path:
    env_root = os.getenv(PERSIST_ROOT_ENV)
    if env_root:
        return Path(env_root)
    if DEFAULT_PERSIST_ROOT.exists():
        return DEFAULT_PERSIST_ROOT
    local_root = Path("./mnt")
    if local_root.exists():
        return local_root.resolve()
    return Path(tempfile.mkdtemp(prefix="purview-persist-"))


def _build_paths(persist_root: Path) -> ModelStoragePaths:
    return ModelStoragePaths(
        persist_root=persist_root,
        models_dir=persist_root / "models",
        spacy_dir=persist_root / "spacy",
        hf_dir=persist_root / "huggingface",
        flair_dir=persist_root / "flair",
        stanza_dir=persist_root / "stanza",
        xdg_cache_dir=persist_root / "xdg-cache",
        torch_dir=persist_root / "torch",
    )


def get_storage_paths() -> ModelStoragePaths:
    global _storage_paths
    if _storage_paths is None:
        _storage_paths = _build_paths(_resolve_persist_root())
    return _storage_paths


def configure_model_caches(paths: ModelStoragePaths) -> None:
    os.environ[PERSIST_ROOT_ENV] = str(paths.persist_root)
    os.environ["XDG_CACHE_HOME"] = str(paths.xdg_cache_dir)
    os.environ["HF_HOME"] = str(paths.hf_dir)
    os.environ["HF_HUB_CACHE"] = str(paths.hf_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(paths.hf_dir / "transformers")
    os.environ["TORCH_HOME"] = str(paths.torch_dir)
    os.environ["SPACY_HOME"] = str(paths.spacy_dir)
    os.environ["STANZA_RESOURCES_DIR"] = str(paths.stanza_dir)
    os.environ["FLAIR_CACHE_ROOT"] = str(paths.flair_dir)
    os.environ["FLAIR_CACHE_DIR"] = str(paths.flair_dir)

    logger.info(
        "Model cache roots set to persist root %s (spacy=%s, hf=%s, flair=%s, stanza=%s)",
        paths.persist_root,
        paths.spacy_dir,
        paths.hf_dir,
        paths.flair_dir,
        paths.stanza_dir,
    )


def ensure_directories(paths: ModelStoragePaths) -> None:
    for path in (
        paths.persist_root,
        paths.models_dir,
        paths.spacy_dir,
        paths.hf_dir,
        paths.flair_dir,
        paths.stanza_dir,
        paths.xdg_cache_dir,
        paths.torch_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def verify_mnt_writable(paths: ModelStoragePaths) -> None:
    verify_env = os.getenv(VERIFY_MNT_WRITABLE_ENV)
    if verify_env is None:
        verify_enabled = not paths.is_ephemeral
    else:
        verify_enabled = verify_env.lower() in {"1", "true", "yes", "on"}
    if not verify_enabled or paths.is_ephemeral:
        logger.info(
            "Skipping persist mount write verification (enabled=%s, ephemeral=%s)",
            verify_enabled,
            paths.is_ephemeral,
        )
        return

    try:
        token = uuid.uuid4().hex
        file_path = paths.persist_root / "testfile.txt"
        file_path.write_text(f"purview-storage-check {token}\n")
        nested_dir = paths.persist_root / "testdir"
        nested_dir.mkdir(parents=True, exist_ok=True)
        nested_file = nested_dir / "testfile.txt"
        nested_file.write_text(f"purview-storage-check {token}\n")
        logger.info("Persist mount write verification succeeded for %s", paths.persist_root)
    except Exception as exc:  # pragma: no cover - filesystem dependent
        raise RuntimeError(
            f"Persist mount {paths.persist_root} is not writable: {exc}"
        ) from exc


def cleanup_legacy_model_dirs(paths: ModelStoragePaths) -> None:
    clean_env = os.getenv(CLEAN_LEGACY_ENV, "1").lower()
    clean_enabled = clean_env in {"1", "true", "yes", "on"}
    if not clean_enabled or paths.is_ephemeral:
        logger.info(
            "Skipping legacy cache cleanup (enabled=%s, ephemeral=%s)",
            clean_enabled,
            paths.is_ephemeral,
        )
        return

    allowlist_dirs = [
        Path("/root/.cache/huggingface"),
        Path("/root/.cache/torch"),
        Path("/root/.cache/transformers"),
        Path("/root/.cache/spacy"),
        Path("/root/.cache/stanza"),
        Path("/root/.cache/flair"),
        Path("/root/.flair"),
        Path("/root/stanza_resources"),
    ]

    for target in allowlist_dirs:
        if not target.exists():
            logger.info("Legacy path not present: %s", target)
            continue
        if target.is_file():
            logger.info("Skipping legacy path (not a directory): %s", target)
            continue
        try:
            shutil.rmtree(target)
            logger.info("Removed legacy model cache: %s", target)
        except Exception as exc:
            logger.warning(
                "Failed to remove legacy model cache %s: %s", target, exc
            )


def init_model_storage() -> ModelStoragePaths:
    global _initialized
    paths = get_storage_paths()
    if _initialized:
        return paths
    if paths.is_ephemeral:
        logger.warning(
            "Persist root %s not found. Using ephemeral storage; models will not persist.",
            DEFAULT_PERSIST_ROOT,
        )
    ensure_directories(paths)
    configure_model_caches(paths)
    verify_mnt_writable(paths)
    cleanup_legacy_model_dirs(paths)
    _initialized = True
    return paths


def resolve_spacy_model_path(model_id: str) -> str:
    paths = get_storage_paths()
    candidate = paths.spacy_dir / model_id
    if candidate.exists():
        return str(candidate)
    return model_id


def model_exists(model_id: str, kind: str) -> bool:
    if not model_id:
        return False
    model_path = Path(model_id)
    if model_path.exists():
        return True

    paths = get_storage_paths()
    kind = kind.lower()

    if kind == "spacy":
        candidate = paths.spacy_dir / model_id
        if (candidate / "meta.json").exists() or (candidate / "config.cfg").exists():
            return True
        try:
            import spacy.util

            return spacy.util.is_package(model_id)
        except Exception:
            return False

    if kind in {"huggingface", "transformers"}:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                cache_dir=str(paths.hf_dir),
                local_files_only=True,
            )
            return True
        except Exception:
            return False

    if kind == "stanza":
        stanza_dir = paths.stanza_dir
        resources = stanza_dir / "resources.json"
        model_dir = stanza_dir / model_id
        if resources.exists() and model_dir.exists():
            if any(model_dir.rglob("*.pt")):
                return True
        return False

    if kind == "flair":
        flair_dir = paths.flair_dir
        slug = model_id.split("/")[-1]
        candidates = [
            flair_dir / f"{slug}.pt",
            flair_dir / f"{slug}.bin",
            flair_dir / "models" / f"{slug}.pt",
            flair_dir / "models" / f"{slug}.bin",
        ]
        if any(candidate.exists() for candidate in candidates):
            return True
        if (flair_dir / "models").exists():
            for item in (flair_dir / "models").iterdir():
                if item.is_file() and slug in item.name and item.suffix in {".pt", ".bin"}:
                    return True
        return False

    return False


def ensure_model(
    model_id: str,
    kind: str,
    downloader_fn: Callable[..., None],
    *args,
    **kwargs,
) -> bool:
    if model_exists(model_id, kind):
        logger.info("Model already present (%s): %s", kind, model_id)
        return False
    logger.info("Model missing (%s): %s. Downloading.", kind, model_id)
    downloader_fn(*args, **kwargs)
    return True


def ensure_and_load_model(
    model_id: str,
    kind: str,
    loader_fn: Callable[..., object],
    *args,
    **kwargs,
):
    if model_exists(model_id, kind):
        logger.info("Model already present (%s): %s", kind, model_id)
    else:
        logger.info("Model missing (%s): %s. Downloading.", kind, model_id)
    return loader_fn(*args, **kwargs)


def ensure_spacy_model(model_id: str) -> None:
    def _download() -> None:
        import spacy

        spacy.cli.download(model_id)
        _copy_spacy_model_to_persist(model_id)

    ensure_model(model_id, "spacy", _download)
    _copy_spacy_model_to_persist(model_id)


def _copy_spacy_model_to_persist(model_id: str) -> None:
    paths = get_storage_paths()
    target = paths.spacy_dir / model_id
    if target.exists():
        return
    try:
        import spacy.util

        package_path = spacy.util.get_package_path(model_id)
    except Exception as exc:
        logger.debug("Unable to resolve spaCy package path for %s: %s", model_id, exc)
        return

    try:
        shutil.copytree(package_path, target, dirs_exist_ok=True)
        logger.info("Copied spaCy model %s to %s", model_id, target)
    except Exception as exc:
        logger.warning("Failed to copy spaCy model %s to %s: %s", model_id, target, exc)
