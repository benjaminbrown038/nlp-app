from typing import List, Dict


def normalize_text(text: str) -> str:
    """
    Normalize whitespace by stripping leading/trailing spaces and collapsing
    multiple spaces into a single space.
    """
    return " ".join(text.strip().split())


def top_k(probs: List[float], labels: List[str], k: int = 3) -> List[Dict]:
    """
    Return the top-k label/score pairs sorted by probability.

    Args:
        probs (List[float]): List of probabilities.
        labels (List[str]): Corresponding labels.
        k (int): Number of top predictions to return.

    Returns:
        List[Dict]: List of {"label": str, "score": float} dictionaries.
    """
    pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    return [{"label": label, "score": float(score)} for label, score in pairs[:k]]
