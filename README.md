---
title: Presidio Demo
emoji: 🅿
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Entities database

This app can import Presidio predefined recognizers into a local SQLite database and render them in the **Entities** UI page.

### Defaults

- `DATABASE_URL`: `sqlite:///./data/app.db`
- `PREDEFINED_RECOGNIZERS_PATH`: `./predefined_recognizers`
- `PURVIEW_PERSIST_ROOT`: `/mnt` (persistent volume mount for ML model caches)
- `VERIFY_MNT_WRITABLE`: `1` (fail fast if persist root is not writable)
- `PURVIEW_CLEAN_LEGACY_MODEL_DIRS`: `1` (remove legacy caches under `/root`)

### Import entities

```
python -m app.import_entities
```

or

```
python scripts/import_entities.py
```

### Run API (optional)

```
uvicorn app.api:app --reload
```

### Streamlit UI

```
streamlit run presidio_streamlit.py
```

## Persistent model storage

This app expects a writable SMB volume mounted at `/mnt` when running in Azure Container Apps.
Mount your Azure Files/SMB volume to `/mnt` so model caches persist across restarts.
Model downloads and caches (spaCy, Hugging Face, Flair, Stanza, Torch, XDG caches) are redirected to subfolders under `/mnt`.

### Verify storage mount

```
python -m purviewanalyzer.verify_storage
```

## Local docs pages

The app renders local documentation pages from `content/docs/` (Code, Tutorial, Installation, FAQ).
Admonitions using `!!! note` and `!!! warning` are supported and rendered as styled callouts.

### Sync docs content

```
python scripts/sync_docs.py
```

Optional: sync a single page:

```
python scripts/sync_docs.py --page faq
```

The sync script pulls content from the official Presidio docs and stores Markdown files in `content/docs/`. After syncing once, the app can render docs without internet access.
