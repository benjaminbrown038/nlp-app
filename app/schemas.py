from pydantic import BaseModel, Field
from typing import List, Optional


class HealthResponse(BaseModel):
status: str = "ok"


class PredictRequest(BaseModel):
text: str = Field(..., min_length=1, description="Input text")
task: str = Field("sentiment", description="sentiment | ner | zero-shot")
candidate_labels: Optional[List[str]] = None # used for zero-shot


class PredictResponse(BaseModel):
task: str
result: dict