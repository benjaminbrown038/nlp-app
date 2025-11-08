"""
Lazy-loading wrappers around HuggingFace transformers pipelines.

These functions only import transformers when needed, so the base
environment can run without installing heavy dependencies.
"""

from typing import Dict, List
from functools import lru_cache


# ---------------------------------------------------------------------
# Lazy import to avoid heavyweight transformer deps at startup
# ---------------------------------------------------------------------
def _lazy_imports():
    from transformers import pipeline  # type: ignore
    return pipeline


# ---------------------------------------------------------------------
# Cached pipeline loaders
# ---------------------------------------------------------------------
@lru_cache()
def get_sentiment():
    pipeline = _lazy_imports()
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@lru_cache()
def get_ner():
    pipeline = _lazy_imports()
    return pipeline(
        "ner",
        model="dslim/bert-base-NER",
        grouped_entities=True
    )


@lru_cache()
def get_zero_shot():
    pipeline = _lazy_imports()
    # Note: smaller MNLI alternatives exist; BART-large MNLI is a good default
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )


# ---------------------------------------------------------------------
# Prediction wrappers
# ---------------------------------------------------------------------
def predict_sentiment(text: str) -> Dict:
    """Run transformer-based sentiment classification."""
    out = get_sentiment()(text)[0]
    return {
        "label": out["label"],
        "score": float(out["score"])
    }


def predict_ner(text: str) -> Dict:
    """Run transformer-based named entity recognition."""
    items = get_ner()(text)
    return {
        "entities": [
            {
                "word": x.get("word"),
                "entity_group": x.get("entity_group"),
                "score": float(x.get("score", 0.0)),
                "start": x.get("start"),
                "end": x.get("end"),
            }
            for x in items
        ]
    }


def predict_zero_shot(text: str, candidate_labels: List[str]) -> Dict:
    """Run transformer-based zero-shot classification."""
    out = get_zero_shot()(text, candidate_labels)
    return {
        "labels": out["labels"],
        "scores": [float(s) for s in out["scores"]],
    }
