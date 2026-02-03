"""Streamlit app for Purview Analyser."""
import logging
import os
import traceback
from pathlib import Path

import dotenv
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from annotated_text import annotated_text
from sqlalchemy import distinct, select
from streamlit_tags import st_tags

from app.config import auto_import_enabled, get_predefined_recognizers_path
from app.db import SessionLocal, init_db
from app.docs_renderer import DOCS_ROOT, render_docs_page
from app.entities_service import get_entity_detail, list_entities
from app.import_entities import load_entity_specs, upsert_entities
from app.models import Entity
from openai_fake_data_generator import OpenAIParams
from presidio_helpers import (
    get_supported_entities,
    analyze,
    anonymize,
    annotate,
    create_fake_data,
    analyzer_engine,
)

st.set_page_config(
    page_title="Purview Analyser",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/",
    },
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")


allow_other_models = os.getenv("ALLOW_OTHER_MODELS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _ensure_entities_loaded() -> None:
    init_db()
    with SessionLocal() as session:
        count = session.execute(select(Entity.id)).first()
    if count is None or auto_import_enabled():
        specs = load_entity_specs(get_predefined_recognizers_path(), import_strategy="auto")
        upsert_entities(specs)


def _get_filter_options() -> tuple[list[str], list[str], list[str]]:
    with SessionLocal() as session:
        languages = [
            row[0]
            for row in session.execute(select(distinct(Entity.language)))
            if row[0]
        ]
        entity_types = [
            row[0]
            for row in session.execute(select(distinct(Entity.entity_type)))
            if row[0]
        ]
        recognizer_types = [
            row[0]
            for row in session.execute(select(distinct(Entity.recognizer_type)))
            if row[0]
        ]
    return sorted(languages), sorted(entity_types), sorted(recognizer_types)


def render_entities() -> None:
    st.title("Entities")
    st.caption("Entities imported from predefined recognizers and stored in the local database.")

    _ensure_entities_loaded()

    col_filters, col_actions = st.columns([3, 1])
    with col_filters:
        search = st.text_input("Search", value="")
        languages, entity_types, recognizer_types = _get_filter_options()
        language = st.selectbox("Language", ["All"] + languages, index=0)
        entity_type = st.selectbox("Entity type", ["All"] + entity_types, index=0)
        recognizer_type = st.selectbox(
            "Recognizer type", ["All"] + recognizer_types, index=0
        )
        enabled = st.selectbox("Enabled", ["All", "Enabled", "Disabled"], index=0)

    with col_actions:
        if st.button("Re-import entities"):
            specs = load_entity_specs(get_predefined_recognizers_path(), import_strategy="auto")
            upsert_entities(specs)
            st.success("Entities imported.")

    enabled_filter = None
    if enabled == "Enabled":
        enabled_filter = True
    elif enabled == "Disabled":
        enabled_filter = False

    with SessionLocal() as session:
        results = list_entities(
            session,
            search=search or None,
            language=None if language == "All" else language,
            entity_type=None if entity_type == "All" else entity_type,
            recognizer_type=None if recognizer_type == "All" else recognizer_type,
            enabled=enabled_filter,
            page=1,
            page_size=500,
        )

    if not results["items"]:
        st.warning("No entities found.")
        return

    st.subheader("Entities")
    st.dataframe(results["items"], use_container_width=True)

    entity_lookup = {
        f"{item['name']} ({item['entity_type']})": item["id"]
        for item in results["items"]
    }
    selected_label = st.selectbox("View details", list(entity_lookup.keys()))
    entity_id = entity_lookup[selected_label]

    with SessionLocal() as session:
        detail = get_entity_detail(session, entity_id)

    if not detail:
        st.error("Entity not found.")
        return

    st.subheader("Details")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.write(f"**Name:** {detail['name']}")
        st.write(f"**Entity type:** {detail['entity_type']}")
        st.write(f"**Language:** {detail['language'] or 'n/a'}")
        st.write(f"**Recognizer type:** {detail['recognizer_type']}")
    with detail_col2:
        st.write(f"**Source file:** {detail['source_file']}")
        st.write(f"**Updated:** {detail['updated_at']}")
        st.write(f"**Enabled:** {detail['enabled']}")

    if detail.get("description"):
        st.markdown(detail["description"])

    st.subheader("Patterns")
    if detail["patterns"]:
        st.dataframe(detail["patterns"], use_container_width=True)
    else:
        st.info("No regex patterns available for this recognizer.")

    st.subheader("Context terms")
    if detail["context"]:
        st.write(", ".join(detail["context"]))
    else:
        st.info("No context terms available.")

    if detail.get("metadata"):
        st.subheader("Metadata")
        st.json(detail["metadata"])


MENU_ITEMS = ["App", "Entities", "Code", "Tutorial", "Installation", "FAQ"]


def _doc_entry(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix:
        index_candidate = path.with_suffix("") / "index.md"
    else:
        index_candidate = path / "index.md"
    return index_candidate if index_candidate.exists() else path


DOCS_PAGES = {
    "Code": _doc_entry(DOCS_ROOT / "code.md"),
    "Tutorial": _doc_entry(DOCS_ROOT / "tutorial.md"),
    "Installation": _doc_entry(DOCS_ROOT / "installation.md"),
    "FAQ": _doc_entry(DOCS_ROOT / "faq.md"),
}
DOCS_BY_REL = {
    path.relative_to(DOCS_ROOT).as_posix(): name for name, path in DOCS_PAGES.items()
}


def _get_query_param(name: str) -> str | None:
    params = st.query_params
    value = params.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _set_query_params(page: str, doc: str | None = None) -> None:
    st.query_params["page"] = page
    if doc:
        st.query_params["doc"] = doc
    else:
        st.query_params.pop("doc", None)


def _resolve_doc_path(doc_param: str | None) -> Path | None:
    if not doc_param:
        return None
    doc_path = (DOCS_ROOT / doc_param).resolve()
    try:
        doc_path.relative_to(DOCS_ROOT.resolve())
    except ValueError:
        return None
    return doc_path if doc_path.exists() else None


def _doc_belongs_to_page(doc_param: str, page: str) -> bool:
    if page == "Tutorial":
        return doc_param == "tutorial.md" or doc_param.startswith("tutorial/")
    if page == "Installation":
        return doc_param == "installation.md" or doc_param.startswith("installation/")
    if page == "FAQ":
        return doc_param == "faq.md" or doc_param.startswith("faq/")
    if page == "Code":
        return doc_param == "code.md" or doc_param.startswith("code/")
    return False


page_param = _get_query_param("page")
doc_param = _get_query_param("doc")
page = page_param if page_param in MENU_ITEMS else None
if not page and doc_param:
    rel = doc_param
    if rel in DOCS_BY_REL:
        page = DOCS_BY_REL[rel]
    elif rel.startswith("tutorial/") or rel.startswith("tutorial"):
        page = "Tutorial"
    elif rel.startswith("installation"):
        page = "Installation"
    elif rel.startswith("faq"):
        page = "FAQ"
    elif rel.startswith("code"):
        page = "Code"
if not page:
    page = st.session_state.get("page", "App")
    if page not in MENU_ITEMS:
        page = "App"

nav_selection = st.radio(
    "Menu",
    MENU_ITEMS,
    index=MENU_ITEMS.index(page),
    horizontal=True,
    label_visibility="collapsed",
    key="page_nav",
)
if nav_selection != page_param:
    doc_target = None
    if nav_selection in DOCS_PAGES:
        doc_target = DOCS_PAGES[nav_selection].relative_to(DOCS_ROOT).as_posix()
    _set_query_params(nav_selection, doc_target)
page = nav_selection

if page == "Entities":
    render_entities()
    st.stop()
if page in DOCS_PAGES:
    doc_path = None
    if doc_param and _doc_belongs_to_page(doc_param, page):
        doc_path = _resolve_doc_path(doc_param)
    doc_path = doc_path or DOCS_PAGES[page]
    render_docs_page(page, doc_path, DOCS_ROOT)
    st.stop()


model_help_text = """
    Select which Named Entity Recognition (NER) model to use for PII detection, in parallel to rule-based recognizers.
    Purview Analyser supports multiple NER packages off-the-shelf, such as spaCy, Huggingface, Stanza and Flair,
    as well as service such as Azure Text Analytics PII.
    """
st_ta_key = st_ta_endpoint = ""

model_list = [
    "spaCy/en_core_web_lg",
    "flair/ner-english-large",
    "HuggingFace/obi/deid_roberta_i2b2",
    "HuggingFace/StanfordAIMI/stanford-deidentifier-base",
    "stanza/en",
    "Azure AI Language",
    "Other",
]
if not allow_other_models:
    model_list.pop()
# Select model
st_model = st.sidebar.selectbox(
    "NER model package",
    model_list,
    index=1,
    help=model_help_text,
)

# Extract model package.
st_model_package = st_model.split("/")[0]

# Remove package prefix (if needed)
st_model = (
    st_model
    if st_model_package.lower() not in ("spacy", "stanza", "huggingface")
    else "/".join(st_model.split("/")[1:])
)

if st_model == "Other":
    st_model_package = st.sidebar.selectbox(
        "NER model OSS package", options=["spaCy", "stanza", "Flair", "HuggingFace"]
    )
    st_model = st.sidebar.text_input(f"NER model name", value="")

if st_model == "Azure AI Language":
    st_ta_key = st.sidebar.text_input(
        f"Azure AI Language key", value=os.getenv("TA_KEY", ""), type="password"
    )
    st_ta_endpoint = st.sidebar.text_input(
        f"Azure AI Language endpoint",
        value=os.getenv("TA_ENDPOINT", default=""),
        help="For more info: https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/personally-identifiable-information/overview",  # noqa: E501
    )


st.sidebar.warning("Note: Models might take some time to download. ")

analyzer_params = (st_model_package, st_model, st_ta_key, st_ta_endpoint)
logger.debug(f"analyzer_params: {analyzer_params}")

st_operator = st.sidebar.selectbox(
    "De-identification approach",
    ["redact", "replace", "synthesize", "highlight", "mask", "hash", "encrypt"],
    index=1,
    help="""
    Select which manipulation to the text is requested after PII has been identified.\n
    - Redact: Completely remove the PII text\n
    - Replace: Replace the PII text with a constant, e.g. <PERSON>\n
    - Synthesize: Replace with fake values (requires an OpenAI key)\n
    - Highlight: Shows the original text with PII highlighted in colors\n
    - Mask: Replaces a requested number of characters with an asterisk (or other mask character)\n
    - Hash: Replaces with the hash of the PII string\n
    - Encrypt: Replaces with an AES encryption of the PII string, allowing the process to be reversed
         """,
)
st_mask_char = "*"
st_number_of_chars = 15
st_encrypt_key = "WmZq4t7w!z%C&F)J"

open_ai_params = None

logger.debug(f"st_operator: {st_operator}")


def set_up_openai_synthesis():
    """Set up the OpenAI API key and model for text synthesis."""

    if os.getenv("OPENAI_TYPE", default="openai") == "Azure":
        openai_api_type = "azure"
        st_openai_api_base = st.sidebar.text_input(
            "Azure OpenAI base URL",
            value=os.getenv("AZURE_OPENAI_ENDPOINT", default=""),
        )
        openai_key = os.getenv("AZURE_OPENAI_KEY", default="")
        st_deployment_id = st.sidebar.text_input(
            "Deployment name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT", default="")
        )
        st_openai_version = st.sidebar.text_input(
            "OpenAI version",
            value=os.getenv("OPENAI_API_VERSION", default="2023-05-15"),
        )
    else:
        openai_api_type = "openai"
        st_openai_version = st_openai_api_base = None
        st_deployment_id = ""
        openai_key = os.getenv("OPENAI_KEY", default="")
    st_openai_key = st.sidebar.text_input(
        "OPENAI_KEY",
        value=openai_key,
        help="See https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key for more info.",
        type="password",
    )
    st_openai_model = st.sidebar.text_input(
        "OpenAI model for text synthesis",
        value=os.getenv("OPENAI_MODEL", default="gpt-3.5-turbo-instruct"),
        help="See more here: https://platform.openai.com/docs/models/",
    )
    return (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    )


if st_operator == "mask":
    st_number_of_chars = st.sidebar.number_input(
        "number of chars", value=st_number_of_chars, min_value=0, max_value=100
    )
    st_mask_char = st.sidebar.text_input(
        "Mask character", value=st_mask_char, max_chars=1
    )
elif st_operator == "encrypt":
    st_encrypt_key = st.sidebar.text_input("AES key", value=st_encrypt_key)
elif st_operator == "synthesize":
    (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    ) = set_up_openai_synthesis()

    open_ai_params = OpenAIParams(
        openai_key=st_openai_key,
        model=st_openai_model,
        api_base=st_openai_api_base,
        deployment_id=st_deployment_id,
        api_version=st_openai_version,
        api_type=openai_api_type,
    )

st_threshold = st.sidebar.slider(
    label="Acceptance threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    help="Define the threshold for accepting a detection as PII. See more here: ",
)

st_return_decision_process = st.sidebar.checkbox(
    "Add analysis explanations to findings",
    value=False,
    help="Add the decision process to the output table. "
    "[Learn more](https://microsoft.github.io/presidio/analyzer/decision_process/).",
)

# Allow and deny lists
st_deny_allow_expander = st.sidebar.expander(
    "Allowlists and denylists",
    expanded=False,
)

with st_deny_allow_expander:
    st_allow_list = st_tags(
        label="Add words to the allowlist", text="Enter word and press enter."
    )
    st.caption(
        "Allowlists contain words that are not considered PII, but are detected as such."
    )

    st_deny_list = st_tags(
        label="Add words to the denylist", text="Enter word and press enter."
    )
    st.caption(
        "Denylists contain words that are considered PII, but are not detected as such."
    )
# Main panel

analyzer_load_state = st.info("Starting Purview Analyser engine...")

analyzer_load_state.empty()

# Read default text
with open("demo_text.txt") as f:
    demo_text = f.readlines()

# Create two columns for before and after
col1, col2 = st.columns(2)

# Before:
col1.subheader("Input")
st_text = col1.text_area(
    label="Enter text", value="".join(demo_text), height=400, key="text_input"
)

try:
    # Choose entities
    st_entities_expander = st.sidebar.expander("Choose entities to look for")
    st_entities = st_entities_expander.multiselect(
        label="Which entities to look for?",
        options=get_supported_entities(*analyzer_params),
        default=list(get_supported_entities(*analyzer_params)),
        help="Limit the list of PII entities detected. "
        "This list is dynamic and based on the NER model and registered recognizers. "
        "[Learn more](https://microsoft.github.io/presidio/analyzer/adding_recognizers/).",
    )

    # Before
    analyzer_load_state = st.info("Starting Purview Analyser engine...")
    analyzer = analyzer_engine(*analyzer_params)
    analyzer_load_state.empty()

    st_analyze_results = analyze(
        *analyzer_params,
        text=st_text,
        entities=st_entities,
        language="en",
        score_threshold=st_threshold,
        return_decision_process=st_return_decision_process,
        allow_list=st_allow_list,
        deny_list=st_deny_list,
    )

    # After
    if st_operator not in ("highlight", "synthesize"):
        with col2:
            st.subheader(f"Output")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                mask_char=st_mask_char,
                number_of_chars=st_number_of_chars,
                encrypt_key=st_encrypt_key,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="De-identified", value=st_anonymize_results.text, height=400
            )
    elif st_operator == "synthesize":
        with col2:
            st.subheader(f"OpenAI Generated output")
            fake_data = create_fake_data(
                st_text,
                st_analyze_results,
                open_ai_params,
            )
            st.text_area(label="Synthetic data", value=fake_data, height=400)
    else:
        st.subheader("Highlighted")
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        # annotated_tokens
        annotated_text(*annotated_tokens)

    # table result
    st.subheader(
        "Findings"
        if not st_return_decision_process
        else "Findings with decision factors"
    )
    if st_analyze_results:
        df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
        df["text"] = [st_text[res.start : res.end] for res in st_analyze_results]

        df_subset = df[["entity_type", "text", "start", "end", "score"]].rename(
            {
                "entity_type": "Entity type",
                "text": "Text",
                "start": "Start",
                "end": "End",
                "score": "Confidence",
            },
            axis=1,
        )
        df_subset["Text"] = [st_text[res.start : res.end] for res in st_analyze_results]
        if st_return_decision_process:
            analysis_explanation_df = pd.DataFrame.from_records(
                [r.analysis_explanation.to_dict() for r in st_analyze_results]
            )
            df_subset = pd.concat([df_subset, analysis_explanation_df], axis=1)
        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
    else:
        st.text("No findings")

except Exception as e:
    print(e)
    traceback.print_exc()
    st.error(e)
