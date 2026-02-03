from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import streamlit as st

DOCS_ROOT = Path("content/docs")


def _slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower()).strip()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug


def _build_toc(markdown_text: str) -> str:
    lines = []
    for line in markdown_text.splitlines():
        if not line.startswith("#"):
            continue
        level = len(line) - len(line.lstrip("#"))
        if level < 2 or level > 3:
            continue
        title = line.lstrip("#").strip()
        anchor = _slugify(title)
        lines.append("  " * (level - 2) + f"- [{title}](#{anchor})")
    return "\n".join(lines)


def _is_external_link(url: str) -> bool:
    scheme = urlparse(url).scheme
    return scheme in {"http", "https", "mailto", "tel"}


def _resolve_local_path(current_doc: Path, target: str) -> Path | None:
    if not target:
        return None
    target = target.lstrip("/")
    candidate = (current_doc.parent / target).resolve()
    docs_root = DOCS_ROOT.resolve()
    try:
        candidate.relative_to(docs_root)
    except ValueError:
        return None

    if candidate.is_dir():
        candidate = candidate / "index.md"

    if not candidate.suffix:
        candidate_with_md = candidate.with_suffix(".md")
        if candidate_with_md.exists():
            candidate = candidate_with_md

    return candidate if candidate.exists() else None


def _data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _rewrite_markdown(
    markdown_text: str,
    current_doc: Path,
    page: str,
    docs_root: Path,
) -> str:
    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    link_pattern = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")

    def replace_image(match: re.Match) -> str:
        alt_text, url = match.group(1), match.group(2).strip()
        url_parts = url.split(maxsplit=1)
        raw_url = url_parts[0]
        if _is_external_link(raw_url):
            return match.group(0)
        anchor_split = raw_url.split("#", 1)
        resolved = _resolve_local_path(current_doc, anchor_split[0])
        if resolved and resolved.exists():
            return f"![{alt_text}]({_data_uri(resolved)})"
        return match.group(0)

    def replace_link(match: re.Match) -> str:
        text, url = match.group(1), match.group(2).strip()
        url_parts = url.split(maxsplit=1)
        raw_url = url_parts[0]
        if raw_url.startswith("#") or _is_external_link(raw_url):
            return match.group(0)
        if raw_url.startswith("?"):
            return match.group(0)

        if "#" in raw_url:
            path_part, anchor = raw_url.split("#", 1)
        else:
            path_part, anchor = raw_url, ""

        if "?" in path_part:
            path_part = path_part.split("?", 1)[0]

        resolved = _resolve_local_path(current_doc, path_part)
        if resolved:
            rel_path = resolved.relative_to(docs_root).as_posix()
            link = f"?page={quote(page)}&doc={quote(rel_path)}"
            if anchor:
                link = f"{link}#{anchor}"
            return f"[{text}]({link})"
        return match.group(0)

    rewritten = image_pattern.sub(replace_image, markdown_text)
    rewritten = link_pattern.sub(replace_link, rewritten)
    return rewritten


def render_docs_page(title: str, markdown_path: Path, docs_root: Path) -> None:
    st.title(title)

    if not markdown_path.exists():
        st.error(f"Missing content file: {markdown_path}")
        return

    markdown_text = markdown_path.read_text(encoding="utf-8")
    toc = _build_toc(markdown_text)
    if toc:
        with st.expander("On this page", expanded=False):
            st.markdown(toc)

    rewritten = _rewrite_markdown(markdown_text, markdown_path, title, docs_root)
    st.markdown(rewritten, unsafe_allow_html=False)
