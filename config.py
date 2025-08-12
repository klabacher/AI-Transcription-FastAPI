import torch
from enum import Enum
import os

# --- CONFIGURAÇÕES GERAIS ---
# Ative o modo debug setando a variável de ambiente DEBUG=true
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
JOB_RETENTION_TIME_SECONDS = 3600  # 1 hora
JANITOR_SLEEP_INTERVAL_SECONDS = 300 # 5 minutos

# --- OPÇÕES DE CONTROLE DA API ---
class DeviceChoice(str, Enum):
    AUTO = "AUTOMATICO"
    CPU = "CPU"
    GPU = "GPU"

class Language(str, Enum):
    PORTUGUESE = "pt"
    ENGLISH = "en" # Mantido para flexibilidade futura

# --- REGISTRO DE MODELOS DISPONÍVEIS ---
# Foco em faster-whisper e modelos com ótimo suporte a PT-BR.
# "workers": 1 -> Define quantos processos workers serão iniciados para este modelo.
# "req_gpu": True -> Indica que o modelo DEVE rodar em GPU.
AVAILABLE_MODELS = {
    # --- Modelos para teste/CPU ---
    "distil_large_v3_ptbr": {
        "impl": "hf_pipeline",
        "model_name": "freds0/distil-whisper-large-v3-ptbr",
        "req_gpu": False,
        "workers": 1,
        "description": "Recomendado para testes locais. Ótima qualidade em PT-BR, leve e rápido em CPU."
    },

    # --- Modelos para produção (requerem GPU) ---
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

    # --- Modelo de produção para CPU/GPU com VRAM limitada ---
    "faster_large-v3_int8": {
        "impl": "faster",
        "model_name": "large-v3",
        "compute_type": "int8",
        "req_gpu": False, # Pode rodar em CPU
        "workers": 1,
        "description": "Qualidade do Large-v3 com menor uso de memória. Ideal para CPUs potentes ou GPUs com VRAM limitada."
    }
}

def get_processing_device(choice: DeviceChoice) -> str:
    """Helper para determinar o dispositivo de processamento."""
    has_gpu = torch.cuda.is_available()
    if choice == DeviceChoice.GPU and not has_gpu:
        raise ValueError("GPU foi solicitada, mas não está disponível ou configurada (CUDA).")
    
    if choice == DeviceChoice.AUTO:
        return 'cuda' if has_gpu else 'cpu'
        
    return choice.value.lower()