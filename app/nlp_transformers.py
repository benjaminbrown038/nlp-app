from typing import Dict, List
from functools import lru_cache


# We import lazily so base install works without transformers


def _lazy_imports():
from transformers import pipeline # type: ignore
return pipeline


@lru_cache()
def get_sentiment() :
pipeline = _lazy_imports()
return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


@lru_cache()
def get_ner() :
pipeline = _lazy_imports()
return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)


@lru_cache()
def get_zero_shot() :
pipeline = _lazy_imports()
# smaller MNLI alternatives exist; this is a common default
return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# wrappers


def predict_sentiment(text: str) -> Dict:
out = get_sentiment()(text)[0]
return {"label": out["label"], "score": float(out["score"])}


def predict_ner(text: str) -> Dict:
items = get_ner()(text)
return {"entities": [
{"word": x.get("word"), "entity_group": x.get("entity_group"), "score": float(x.get("score", 0.0)),
"start": x.get("start"), "end": x.get("end")} for x in items
]}


def predict_zero_shot(text: str, candidate_labels: List[str]) -> Dict:
out = get_zero_shot()(text, candidate_labels)
return {
"labels": out["labels"],
"scores": [float(s) for s in out["scores"]]
}