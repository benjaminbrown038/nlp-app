"""A tiny baseline sentiment model using TF‑IDF + LogisticRegression.
Trains on a miniature toy dataset at import time (milliseconds). Purely illustrative.
"""
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


# tiny toy set
X = [
"I love this! absolutely fantastic",
"This is great and very useful",
"What a wonderful experience",
"I hate this, it is terrible",
"awful and disappointing",
"this is bad"
]
y = [1, 1, 1, 0, 0, 0] # 1=pos, 0=neg


pipe: Pipeline = Pipeline([
("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2))),
("clf", LogisticRegression(max_iter=200))
])
pipe.fit(X, y)


label_map = {0: "NEGATIVE", 1: "POSITIVE"}


def predict_sentiment(text: str) -> Dict:
proba = pipe.predict_proba([text])[0]
idx = int(np.argmax(proba))
return {
"label": label_map[idx],
"score": float(proba[idx])
}