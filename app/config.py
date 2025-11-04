from pydantic import BaseModel
from functools import lru_cache
import os


class Settings(BaseModel):
app_name: str = "nlp-app"
environment: str = os.getenv("ENV", "dev")
enable_transformers: bool = os.getenv("ENABLE_TRANSFORMERS", "0") == "1"


@lru_cache()
def get_settings() -> Settings:
return Settings()