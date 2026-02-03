from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import markdown
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


def _resolve_local_path(current_doc: Path, target: str, docs_root: Path) -> Path | None:
    if not target:
        return None
    target = target.lstrip("/")
    candidate = (current_doc.parent / target).resolve()
    docs_root = docs_root.resolve()
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
    docs_root = docs_root.resolve()
    current_doc = current_doc.resolve()

    def replace_image(match: re.Match) -> str:
        alt_text, url = match.group(1), match.group(2).strip()
        url_parts = url.split(maxsplit=1)
        raw_url = url_parts[0]
        if _is_external_link(raw_url):
            return match.group(0)
        anchor_split = raw_url.split("#", 1)
        resolved = _resolve_local_path(current_doc, anchor_split[0], docs_root)
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

        resolved = _resolve_local_path(current_doc, path_part, docs_root)
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

    if not st.session_state.get("admonition_css_loaded"):
        st.markdown(
            """
<style>
.admonition {
  border: 1px solid;
  border-radius: 6px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  margin: 16px 0;
}
.admonition > .admonition-title {
  margin: 0;
  padding: 10px 12px;
  font-weight: 600;
  color: #111827;
  display: flex;
  align-items: center;
  gap: 8px;
}
.admonition > :not(.admonition-title) {
  padding: 12px;
  margin: 0;
  color: #111827;
  line-height: 1.45;
}
.admonition.note { border-color: #3b82f6; }
.admonition.note > .admonition-title { background: #eff6ff; }
.admonition.note > .admonition-title::before {
  content: "";
  width: 18px;
  height: 18px;
  border-radius: 999px;
  background: #3b82f6;
  display: inline-block;
  flex: 0 0 18px;
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='white' d='M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25Zm2.92 2.83H5v-.92l8.06-8.06.92.92L5.92 20.08ZM20.71 7.04a1.003 1.003 0 0 0 0-1.42L18.37 3.29a1.003 1.003 0 0 0-1.42 0l-1.83 1.83 3.75 3.75 1.84-1.83Z'/%3E%3C/svg%3E") center / 14px 14px no-repeat;
          mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='white' d='M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25Zm2.92 2.83H5v-.92l8.06-8.06.92.92L5.92 20.08ZM20.71 7.04a1.003 1.003 0 0 0 0-1.42L18.37 3.29a1.003 1.003 0 0 0-1.42 0l-1.83 1.83 3.75 3.75 1.84-1.83Z'/%3E%3C/svg%3E") center / 14px 14px no-repeat;
}
.admonition.warning { border-color: #f59e0b; }
.admonition.warning > .admonition-title { background: #fffbeb; }
.admonition.warning > .admonition-title::before {
  content: "";
  width: 18px;
  height: 18px;
  display: inline-block;
  flex: 0 0 18px;
  background: #f59e0b;
  clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
}
</style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state["admonition_css_loaded"] = True

    markdown_text = markdown_path.read_text(encoding="utf-8")
    toc = _build_toc(markdown_text)
    if toc:
        with st.expander("On this page", expanded=False):
            st.markdown(toc)

    rewritten = _rewrite_markdown(markdown_text, markdown_path, title, docs_root)
    html = markdown.markdown(
        rewritten,
        extensions=[
            "admonition",
            "fenced_code",
            "tables",
            "sane_lists",
        ],
    )
    st.markdown(html, unsafe_allow_html=True)
