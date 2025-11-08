"""
A tiny baseline sentiment model using TF-IDF + LogisticRegression.

Trains on a miniature toy dataset at import time (milliseconds).
This is purely illustrative and not intended for real production use.
"""

from typing import Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------
# Tiny toy dataset
# ---------------------------------------------------------------------
X = [
    "I love this! absolutely fantastic",
    "This is great and very useful",
    "What a wonderful experience",
    "I hate this, it is terrible",
    "awful and disappointing",
    "this is bad"
]

y = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative

# ---------------------------------------------------------------------
# Build pipeline: TF-IDF → Logistic Regression
# ---------------------------------------------------------------------
pipe: Pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=200)),
])

# Train immediately at import time
pipe.fit(X, y)

# Label map for outputs
label_map = {0: "NEGATIVE", 1: "POSITIVE"}


# ---------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------
def predict_sentiment(text: str) -> Dict:
    """Predict sentiment for a single input string."""
    proba = pipe.predict_proba([text])[0]
    idx = int(np.argmax(proba))

    return {
        "label": label_map[idx],
        "score": float(proba[idx]),
    }
