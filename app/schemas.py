from pydantic import BaseModel
from typing import List, Optional


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictRequest(BaseModel):
    text: str
    task: str
    candidate_labels: Optional[List[str]] = None


class PredictResponse(BaseModel):
    task: str
    result: dict
