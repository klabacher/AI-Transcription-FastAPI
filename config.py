from enum import Enum
import torch

DEBUG = True
JOB_RETENTION_TIME_SECONDS = 3600
JANITOR_SLEEP_INTERVAL_SECONDS = 300

class DeviceChoice(str, Enum):
    AUTO = "automatic"
    CPU = "cpu"
    GPU = "gpu"

class Language(str, Enum):
    PORTUGUESE = "pt"
    ENGLISH = "en"

# Model registry: apenas faster-whisper e distil-ptbr
AVAILABLE_MODELS = {
    "faster_small_fp16": {
        "impl": "faster",
        "model_name": "small",
        "compute_type": "float16",
        "req_gpu": True
    },
    "distil_ptbr": {
        "impl": "hf_pipeline",
        "model_name": "freds0/distil-whisper-large-v3-ptbr",
        "req_gpu": False
    },
}


def has_gpu() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def get_device(choice: DeviceChoice) -> str:
    if choice == DeviceChoice.GPU:
        if not has_gpu():
            raise RuntimeError("GPU solicitada, mas não disponível")
        return "cuda"
    if choice == DeviceChoice.CPU:
        return "cpu"
    # AUTO
    return "cuda" if has_gpu() else "cpu"


def filter_models_for_device(device: str) -> dict:
    if device == "cpu":
        return {k: v for k, v in AVAILABLE_MODELS.items() if not v.get("req_gpu", False)}
    return AVAILABLE_MODELS
