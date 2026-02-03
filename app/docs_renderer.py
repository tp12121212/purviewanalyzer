from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import quote, urlparse

import markdown
import streamlit as st

DOCS_ROOT = Path("content/docs")
BRAND_NAME = "Purview Analyser"
BRAND_NAME_LOWER = "purview analyser"
SOURCE_NAME = "Presidio"
SOURCE_NAME_LOWER = "presidio"


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


def _inject_heading_anchors(markdown_text: str) -> str:
    lines = []
    for line in markdown_text.splitlines():
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            anchor = _slugify(title)
            lines.append(f'<a id="{anchor}"></a>')
            lines.append(line)
            continue
        lines.append(line)
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


def _convert_admonitions(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    i = 0
    admonition_re = re.compile(r'^(?P<indent>\s*)!!!\s+(?P<kind>note|warning)\s*(?:"(?P<title>[^"]+)")?\s*$')

    while i < len(lines):
        line = lines[i]
        match = admonition_re.match(line)
        if not match:
            output.append(line)
            i += 1
            continue

        indent = match.group("indent")
        kind = match.group("kind")
        title = match.group("title") or kind.capitalize()
        base_indent = indent + "    "
        base_tab = indent + "\t"
        body_lines: list[str] = []
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if next_line.startswith(base_indent):
                body_lines.append(next_line[len(base_indent):])
                i += 1
                continue
            if next_line.startswith(base_tab):
                body_lines.append(next_line[len(base_tab):])
                i += 1
                continue
            if next_line.strip() == "":
                body_lines.append("")
                i += 1
                continue
            break

        body_md = "\n".join(body_lines).strip("\n")
        body_html = markdown.markdown(
            body_md,
            extensions=[
                "fenced_code",
                "tables",
                "sane_lists",
            ],
        )
        output.append(f'{indent}<div class="admonition {kind}">')
        output.append(f'{indent}  <p class="admonition-title">{title}</p>')
        output.append(f"{indent}  {body_html}")
        output.append(f"{indent}</div>")

    return "\n".join(output)


def _extract_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
        if line.startswith("## "):
            return line[3:].strip()
    return fallback


def _strip_leading_title(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip("\n")
    return markdown_text


def _build_tutorial_single_page(index_path: Path) -> str:
    base_dir = index_path.parent
    index_text = index_path.read_text(encoding="utf-8")

    toc_links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", index_text)
    tutorial_files = []
    for link in toc_links:
        if not link.endswith(".md"):
            continue
        candidate = (base_dir / link).resolve()
        if candidate.exists():
            tutorial_files.append(candidate)

    sections = []
    for doc_path in tutorial_files:
        content = doc_path.read_text(encoding="utf-8")
        title = _extract_title(content, doc_path.stem.replace("_", " ").title())
        section_body = _strip_leading_title(content)
        sections.append(f"## {title}\n\n{section_body.strip()}\n")

    combined = index_text.strip() + "\n\n" + "\n".join(sections)

    # Rewrite tutorial TOC links to in-page anchors.
    for doc_path in tutorial_files:
        content = doc_path.read_text(encoding="utf-8")
        title = _extract_title(content, doc_path.stem.replace("_", " ").title())
        anchor = _slugify(title)
        combined = re.sub(
            rf"\((?:{re.escape(doc_path.name)})\)",
            f"(#{anchor})",
            combined,
        )

    return combined


def _strip_leading_toc(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    if not lines:
        return markdown_text

    # Find first heading.
    start = 0
    while start < len(lines) and not lines[start].startswith("#"):
        start += 1
    if start >= len(lines):
        return markdown_text

    i = start + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    if i >= len(lines) or not lines[i].lstrip().startswith(("-", "*")):
        return markdown_text

    # Consume list block (toc) starting here.
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith(("-", "*")) or stripped == "":
            i += 1
            continue
        break

    new_lines = lines[: start + 1] + lines[i:]
    return "\n".join(new_lines)


def _strip_named_toc(markdown_text: str, heading_name: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    i = 0
    heading_re = re.compile(rf"^#+\s+{re.escape(heading_name)}\s*$", re.IGNORECASE)

    while i < len(lines):
        line = lines[i]
        if heading_re.match(line):
            # Skip heading line.
            i += 1
            # Skip blank lines after heading.
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            # Skip list block following the heading.
            while i < len(lines):
                stripped = lines[i].lstrip()
                if stripped.startswith(("-", "*")) or stripped == "":
                    i += 1
                    continue
                break
            continue
        output.append(line)
        i += 1

    return "\n".join(output)


def _apply_branding(markdown_text: str) -> str:
    code_fence_re = re.compile(r"^(```|~~~)")
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    inline_code_re = re.compile(r"`[^`]+`")

    lines = markdown_text.splitlines()
    output: list[str] = []
    in_code_block = False
    fence_token = ""

    def replace_text(text: str) -> str:
        text = re.sub(rf"\b{SOURCE_NAME}\b", BRAND_NAME, text)
        text = re.sub(rf"\b{SOURCE_NAME_LOWER}\b", BRAND_NAME_LOWER, text)
        return text

    def replace_outside_inline_code(text: str) -> str:
        parts: list[str] = []
        last = 0
        for match in inline_code_re.finditer(text):
            parts.append(replace_text(text[last:match.start()]))
            parts.append(match.group(0))
            last = match.end()
        parts.append(replace_text(text[last:]))
        return "".join(parts)

    for line in lines:
        fence_match = code_fence_re.match(line.strip())
        if fence_match:
            token = fence_match.group(1)
            if not in_code_block:
                in_code_block = True
                fence_token = token
            elif fence_token == token:
                in_code_block = False
                fence_token = ""
            output.append(line)
            continue

        if in_code_block:
            output.append(line)
            continue

        rebuilt = []
        last_end = 0
        for match in link_re.finditer(line):
            rebuilt.append(replace_outside_inline_code(line[last_end:match.start()]))
            link_text, link_url = match.group(1), match.group(2)
            rebuilt.append(f"[{replace_outside_inline_code(link_text)}]({link_url})")
            last_end = match.end()
        rebuilt.append(replace_outside_inline_code(line[last_end:]))
        output.append("".join(rebuilt))

    return "\n".join(output)


def render_docs_page(title: str, markdown_path: Path, docs_root: Path) -> None:
    st.title(title)

    if not markdown_path.exists():
        st.error(f"Missing content file: {markdown_path}")
        return

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

    if title == "Tutorial" and markdown_path.name == "index.md":
        markdown_text = _build_tutorial_single_page(markdown_path)
    else:
        markdown_text = markdown_path.read_text(encoding="utf-8")
    markdown_text = _apply_branding(markdown_text)
    markdown_text = _apply_branding(markdown_text)

    if title in {"FAQ", "Tutorial"}:
        markdown_text = _strip_leading_toc(markdown_text)
        markdown_text = _strip_named_toc(markdown_text, "Table of contents")

    toc = _build_toc(markdown_text)
    if toc:
        with st.expander("On this page", expanded=False):
            st.markdown(toc)

    rewritten = _rewrite_markdown(markdown_text, markdown_path, title, docs_root)
    rewritten = _convert_admonitions(rewritten)
    rewritten = _inject_heading_anchors(rewritten)
    st.markdown(rewritten, unsafe_allow_html=True)
