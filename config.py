import torch
from enum import Enum
import os
import logging

DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
JOB_RETENTION_TIME_SECONDS = 3600
JANITOR_SLEEP_INTERVAL_SECONDS = 300

class Language(str, Enum):
    PORTUGUESE = "pt"
    ENGLISH = "en"

AVAILABLE_MODELS = {
    "distil_large_v3_ptbr": {
        "impl": "hf_pipeline",
        "model_name": "freds0/distil-whisper-large-v3-ptbr",
        "req_gpu": False,
        "workers": 1,
        "description": "Recomendado para testes locais. Ótima qualidade em PT-BR, leve e rápido em CPU."
    },
    "faster_medium_fp16": {
        "impl": "faster",
        "model_name": "medium",
        "compute_type": "float16",
        "req_gpu": True,
        "workers": 1,
        "description": "Excelente equilíbrio entre velocidade e qualidade em GPU."
    },
    "faster_large-v3_fp16": {
        "impl": "faster",
        "model_name": "large-v3",
        "compute_type": "float16",
        "req_gpu": True,
        "workers": 1,
        "description": "Máxima qualidade e precisão em PT-BR. Requer GPU potente (VRAM > 8GB)."
    },
    "faster_large-v3_int8": {
        "impl": "faster",
        "model_name": "large-v3",
        "compute_type": "int8",
        "req_gpu": False,
        "workers": 1,
        "description": "Qualidade do Large-v3 com menor uso de memória. Ideal para CPUs potentes ou GPUs com VRAM limitada."
    }
}