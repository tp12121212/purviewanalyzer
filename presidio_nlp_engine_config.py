import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

from app.model_storage import (
    ensure_model,
    ensure_spacy_model,
    get_storage_paths,
    resolve_spacy_model_path,
)
from presidio_analyzer import RecognizerRegistry
from presidio_analyzer.nlp_engine import (
    NlpEngine,
    NlpEngineProvider,
)

logger = logging.getLogger("presidio-streamlit")


def _iter_local_country_specific_recognizer_classes() -> Iterable[type]:
    root = Path(__file__).resolve().parent / "predefined_recognizers" / "country_specific"
    if not root.exists():
        return []

    classes: list[type] = []
    for module_file in root.rglob("*_recognizer.py"):
        module_name = f"local_country_recognizers.{module_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if not spec or not spec.loader:
            continue

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            logger.warning("Failed importing %s: %s", module_file.name, exc)
            continue

        for attr in vars(module).values():
            if not isinstance(attr, type):
                continue
            if attr.__module__ != module_name:
                continue
            if "Recognizer" not in attr.__name__:
                continue
            classes.append(attr)

    return classes


def _load_additional_customer_recognizers(registry: RecognizerRegistry) -> None:
    """Load country-specific/customer recognizers from local source files."""

    from presidio_analyzer import EntityRecognizer

    already_loaded = {rec.__class__.__name__ for rec in registry.recognizers}

    for recognizer_cls in _iter_local_country_specific_recognizer_classes():
        recognizer_name = recognizer_cls.__name__

        if recognizer_name in already_loaded:
            continue

        try:
            recognizer = recognizer_cls()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed loading recognizer %s: %s", recognizer_name, exc)
            continue

        if not isinstance(recognizer, EntityRecognizer):
            continue

        registry.add_recognizer(recognizer)
        already_loaded.add(recognizer_name)


def _ensure_huggingface_model(model_id: str) -> None:
    def _download() -> None:
        from huggingface_hub import snapshot_download

        paths = get_storage_paths()
        snapshot_download(repo_id=model_id, cache_dir=str(paths.hf_dir))

    ensure_model(model_id, "huggingface", _download)


def _ensure_stanza_model(model_id: str) -> None:
    def _download() -> None:
        import stanza

        paths = get_storage_paths()
        stanza.download(model_id, model_dir=str(paths.stanza_dir))

    ensure_model(model_id, "stanza", _download)


def create_nlp_engine_with_spacy(
    model_path: str,
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Instantiate an NlpEngine with a spaCy model
    :param model_path: path to model / model name.
    """
    ensure_spacy_model(model_path)
    resolved_model = resolve_spacy_model_path(model_path)

    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": resolved_model}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "NORP": "NRP",
                "FAC": "FACILITY",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "LOCATION": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ORG", "ORGANIZATION"],
        },
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    _load_additional_customer_recognizers(registry)

    return nlp_engine, registry


def create_nlp_engine_with_stanza(
    model_path: str,
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Instantiate an NlpEngine with a stanza model
    :param model_path: path to model / model name.
    """
    _ensure_stanza_model(model_path)

    nlp_configuration = {
        "nlp_engine_name": "stanza",
        "models": [{"lang_code": "en", "model_name": model_path}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "NORP": "NRP",
                "FAC": "FACILITY",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "LOCATION": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            }
        },
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    _load_additional_customer_recognizers(registry)

    return nlp_engine, registry


def create_nlp_engine_with_transformers(
    model_path: str,
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Instantiate an NlpEngine with a TransformersRecognizer and a small spaCy model.
    The TransformersRecognizer would return results from Transformers models, the spaCy model
    would return NlpArtifacts such as POS and lemmas.
    :param model_path: HuggingFace model path.
    """
    print(f"Loading Transformers model: {model_path} of type {type(model_path)}")
    ensure_spacy_model("en_core_web_sm")
    resolved_spacy = resolve_spacy_model_path("en_core_web_sm")
    _ensure_huggingface_model(model_path)

    nlp_configuration = {
        "nlp_engine_name": "transformers",
        "models": [
            {
                "lang_code": "en",
                "model_name": {
                    "spacy": resolved_spacy,
                    "transformers": model_path,
                },
            }
        ],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "LOC": "LOCATION",
                "LOCATION": "LOCATION",
                "GPE": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "NORP": "NRP",
                "AGE": "AGE",
                "ID": "ID",
                "EMAIL": "EMAIL",
                "PATIENT": "PERSON",
                "STAFF": "PERSON",
                "HOSP": "ORGANIZATION",
                "PATORG": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
                "PHONE": "PHONE_NUMBER",
                "HCW": "PERSON",
                "HOSPITAL": "ORGANIZATION",
                "FACILITY": "LOCATION",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ID"],
            "labels_to_ignore": [
                "CARDINAL",
                "EVENT",
                "LANGUAGE",
                "LAW",
                "MONEY",
                "ORDINAL",
                "PERCENT",
                "PRODUCT",
                "QUANTITY",
                "WORK_OF_ART",
            ],
        },
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)
    _load_additional_customer_recognizers(registry)

    return nlp_engine, registry


def create_nlp_engine_with_flair(
    model_path: str,
) -> Tuple[NlpEngine, RecognizerRegistry]:
    """
    Instantiate an NlpEngine with a FlairRecognizer and a small spaCy model.
    The FlairRecognizer would return results from Flair models, the spaCy model
    would return NlpArtifacts such as POS and lemmas.
    :param model_path: Flair model path.
    """
    from flair_recognizer import FlairRecognizer

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    _load_additional_customer_recognizers(registry)

    # there is no official Flair NlpEngine, hence we load it as an additional recognizer

    ensure_spacy_model("en_core_web_sm")
    # Using a small spaCy model + a Flair NER model
    flair_recognizer = FlairRecognizer(model_path=model_path)
    resolved_spacy = resolve_spacy_model_path("en_core_web_sm")
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": resolved_spacy}],
    }
    registry.add_recognizer(flair_recognizer)
    registry.remove_recognizer("SpacyRecognizer")

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    return nlp_engine, registry


def create_nlp_engine_with_azure_ai_language(ta_key: str, ta_endpoint: str):
    """
    Instantiate an NlpEngine with a TextAnalyticsWrapper and a small spaCy model.
    The TextAnalyticsWrapper would return results from calling Azure Text Analytics PII, the spaCy model
    would return NlpArtifacts such as POS and lemmas.
    :param ta_key: Azure Text Analytics key.
    :param ta_endpoint: Azure Text Analytics endpoint.
    """
    from azure_ai_language_wrapper import AzureAIServiceWrapper

    if not ta_key or not ta_endpoint:
        raise RuntimeError("Please fill in the Text Analytics endpoint details")

    ensure_spacy_model("en_core_web_sm")
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    _load_additional_customer_recognizers(registry)

    azure_ai_language_recognizer = AzureAIServiceWrapper(
        ta_endpoint=ta_endpoint, ta_key=ta_key
    )
    resolved_spacy = resolve_spacy_model_path("en_core_web_sm")
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": resolved_spacy}],
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    registry.add_recognizer(azure_ai_language_recognizer)
    registry.remove_recognizer("SpacyRecognizer")

    return nlp_engine, registry
