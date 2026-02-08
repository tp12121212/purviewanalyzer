from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st


@dataclass
class RepoEntry:
    name: str
    path: str
    type: str
    size: int | None = None


@dataclass
class RepoFile:
    path: str
    content: str
    encoding: str
    size: int


def _github_api_get(url: str, timeout: int = 15) -> dict[str, Any] | list[dict[str, Any]]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False, ttl=300)
def list_repo_entries(owner: str, repo: str, path: str = "", ref: str = "main") -> list[RepoEntry]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    payload = _github_api_get(url)
    if isinstance(payload, dict) and payload.get("type") == "file":
        return [RepoEntry(name=payload["name"], path=payload["path"], type="file", size=payload.get("size"))]
    entries = []
    for item in payload:
        entries.append(
            RepoEntry(
                name=item.get("name", ""),
                path=item.get("path", ""),
                type=item.get("type", ""),
                size=item.get("size"),
            )
        )
    return sorted(entries, key=lambda e: (e.type != "dir", e.name.lower()))


@st.cache_data(show_spinner=False, ttl=300)
def fetch_repo_file(owner: str, repo: str, path: str, ref: str = "main") -> RepoFile:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    payload = _github_api_get(url)
    if payload.get("type") != "file":
        raise ValueError("Not a file")
    content = payload.get("content", "")
    encoding = payload.get("encoding", "base64")
    size = payload.get("size", 0)
    if encoding == "base64":
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
    else:
        decoded = content
    return RepoFile(path=path, content=decoded, encoding=encoding, size=size)


def render_repo_browser(owner: str, repo: str, ref: str = "main") -> None:
    st.subheader("Repository")

    if "repo_path" not in st.session_state:
        st.session_state.repo_path = ""
    if "repo_file" not in st.session_state:
        st.session_state.repo_file = None

    path = st.session_state.repo_path

    try:
        entries = list_repo_entries(owner, repo, path, ref)
    except requests.HTTPError as exc:
        st.error(f"Failed to load repository contents: {exc}")
        return

    breadcrumb_parts = [p for p in path.split("/") if p]
    col_breadcrumb, col_reset = st.columns([4, 1])
    with col_breadcrumb:
        if st.button("/"):
            st.session_state.repo_path = ""
            st.session_state.repo_file = None
            st.rerun()
        for i, part in enumerate(breadcrumb_parts):
            if st.button(part, key=f"crumb-{i}"):
                st.session_state.repo_path = "/".join(breadcrumb_parts[: i + 1])
                st.session_state.repo_file = None
                st.rerun()
    with col_reset:
        if st.button("Reset", key="repo-reset"):
            st.session_state.repo_path = ""
            st.session_state.repo_file = None
            st.rerun()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        if path:
            parent = "/".join(path.split("/")[:-1])
            if st.button("..", key="repo-up"):
                st.session_state.repo_path = parent
                st.session_state.repo_file = None
                st.rerun()

        for entry in entries:
            if entry.type == "dir":
                if st.button(f"📁 {entry.name}", key=f"dir-{entry.path}"):
                    st.session_state.repo_path = entry.path
                    st.session_state.repo_file = None
                    st.rerun()
            elif entry.type == "file":
                if st.button(f"📄 {entry.name}", key=f"file-{entry.path}"):
                    st.session_state.repo_file = entry.path
                    st.rerun()

    with col_right:
        selected = st.session_state.repo_file
        if not selected:
            st.info("Select a file to preview its contents.")
            return
        try:
            file_data = fetch_repo_file(owner, repo, selected, ref)
        except requests.HTTPError as exc:
            st.error(f"Failed to load file: {exc}")
            return
        if file_data.size > 200_000:
            st.warning("File is large; showing first 200KB.")
            content = file_data.content[:200_000]
        else:
            content = file_data.content
        st.code(content, language="")
