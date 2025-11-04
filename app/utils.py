from typing import List


def normalize_text(text: str) -> str:
return " ".join(text.strip().split())


# simple label formatting
def top_k(probs: List[float], labels: List[str], k: int = 3):
pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
return [{"label": l, "score": float(p)} for l, p in pairs[:k]]