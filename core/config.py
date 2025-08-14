from pydantic_settings import BaseSettings
from pydantic import SecretStr
from typing import Literal
from enum import Enum


class Language(str, Enum):
    """
    Enumeration for the supported languages.
    """

    PORTUGUESE = "pt"
    ENGLISH = "en"


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables.
    Utilizes pydantic-settings for type validation and loading from .env files.
    """

    DEBUG: bool = True
    EXECUTION_BACKEND: Literal["local", "distributed"] = "local"
    REDIS_URL: str = "redis://localhost:6379/0"
    API_KEY: SecretStr = "your-secret-api-key"  # This should be set in the environment

    # --- Job Lifecycle Settings ---
    JOB_RETENTION_TIME_SECONDS: int = 3600  # 1 hour
    JANITOR_SLEEP_INTERVAL_SECONDS: int = 300  # 5 minutes

    class Config:
        # This allows loading variables from a .env file
        env_file = ".env"
        env_file_encoding = "utf-8"


# --- Model Configuration ---
# This dictionary contains the configuration for all available transcription models.
AVAILABLE_MODELS = {
    "distil_large_v3_ptbr": {
        "impl": "hf_pipeline",
        "model_name": "freds0/distil-whisper-large-v3-ptbr",
        "req_gpu": False,
        "workers": 1,
        "description": "Recommended for local testing. Great quality in PT-BR, lightweight and fast on CPU.",
    },
    "faster_medium_fp16": {
        "impl": "faster",
        "model_name": "medium",
        "compute_type": "float16",
        "req_gpu": True,
        "workers": 1,
        "description": "Excellent balance between speed and quality on GPU.",
    },
    "faster_large-v3_fp16": {
        "impl": "faster",
        "model_name": "large-v3",
        "compute_type": "float16",
        "req_gpu": True,
        "workers": 1,
        "description": "Maximum quality and precision in PT-BR. Requires a powerful GPU (VRAM > 8GB).",
    },
    "faster_large-v3_int8": {
        "impl": "faster",
        "model_name": "large-v3",
        "compute_type": "int8",
        "req_gpu": False,
        "workers": 1,
        "description": "Quality of Large-v3 with lower memory usage. Ideal for powerful CPUs or GPUs with limited VRAM.",
    },
}
