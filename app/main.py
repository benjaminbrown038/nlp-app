from fastapi import FastAPI, HTTPException
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from app.config import get_settings
from app.utils import normalize_text
from app import nlp_baseline
from app import nlp_transformers

app = FastAPI(title="nlp-app")

# Optional transformers (enabled via env var)
try:
    _has_transformers = True
except Exception:
    _has_transformers = False


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    settings = get_settings()
    text = normalize_text(req.text)

    # -----------------------
    # Sentiment classification
    # -----------------------
    if req.task == "sentiment":
        if settings.enable_transformers and _has_transformers:
            result = nlp_transformers.predict_sentiment(text)
        else:
            result = nlp_baseline.predict_sentiment(text)

        return PredictResponse(task="sentiment", result=result)

    # -----------------------
    # Named Entity Recognition
    # -----------------------
    elif req.task == "ner":
        if settings.enable_transformers and _has_transformers:
            result = nlp_transformers.predict_ner(text)
            return PredictResponse(task="ner", result=result)

        raise HTTPException(
            status_code=400,
            detail="NER requires transformers. Set ENABLE_TRANSFORMERS=1 and install optional deps."
        )

    # -----------------------
    # Zero-Shot Classification
    # -----------------------
    elif req.task == "zero-shot":
        if not req.candidate_labels:
            raise HTTPException(
                status_code=400,
                detail="candidate_labels required for zero-shot"
            )

        if settings.enable_transformers and _has_transformers:
            result = nlp_transformers.predict_zero_shot(text, req.candidate_labels)
            return PredictResponse(task="zero-shot", result=result)

        raise HTTPException(
            status_code=400,
            detail="Zero-shot requires transformers. Set ENABLE_TRANSFORMERS=1 and install optional deps."
        )

    # -----------------------
    # Unknown task
    # -----------------------
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'"
        )
