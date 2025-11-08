from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "nlp-app"
    enable_transformers: bool = False  # read from ENABLE_TRANSFORMERS env var

    model_config = {
        "env_prefix": "",   # read root-level env vars
        "env_file": ".env",
        "case_sensitive": False
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
