import torch
from enum import Enum

# --- CONFIGURAÇÕES DE CACHE E LIMPEZA ---
# Tempo (em segundos) que um job concluído ou falho deve ser mantido na memória antes de ser apagado.
# Padrão: 3600 segundos = 1 hora
JOB_RETENTION_TIME_SECONDS = 3600

# Intervalo (em segundos) em que o processo de limpeza (janitor) irá rodar para verificar jobs antigos.
# Padrão: 300 segundos = 5 minutos
JANITOR_SLEEP_INTERVAL_SECONDS = 300

# --- OPÇÕES DE CONTROLE ---
class DeviceChoice(str, Enum):
    AUTO = "AUTOMATICO"
    CPU = "CPU"
    GPU = "GPU"

class Language(str, Enum):
    PORTUGUESE = "pt"
    ENGLISH = "en"

# --- MODELOS DISPONÍVEIS ---
AVAILABLE_MODELS = {
    'assemblyai_best': {'impl': 'assemblyai', 'model_name': 'best', 'req_gpu': False},
    'openai_small': {'impl': 'openai', 'model_name': 'small', 'req_gpu': False},
    'openai_medium': {'impl': 'openai', 'model_name': 'medium', 'req_gpu': False},
    'openai_large-v3': {'impl': 'openai', 'model_name': 'large-v3', 'req_gpu': False},
    'faster_small_fp16': {'impl': 'faster', 'model_name': 'small', 'compute_type': 'float16', 'req_gpu': True},
    'faster_medium_fp16': {'impl': 'faster', 'model_name': 'medium', 'compute_type': 'float16', 'req_gpu': True},
    'faster_large-v3_fp16': {'impl': 'faster', 'model_name': 'large-v3', 'compute_type': 'float16', 'req_gpu': True},
    'faster_large-v3_int8': {'impl': 'faster', 'model_name': 'large-v3', 'compute_type': 'int8', 'req_gpu': False},
    'distil_large-v2': {'impl': 'hf_pipeline', 'model_name': 'distil-whisper/distil-large-v2', 'req_gpu': False},
    'distil_large-v3_pt-br_freds0': {'impl': 'hf_pipeline', 'model_name': 'freds0/distil-whisper-large-v3-ptbr', 'req_gpu': False},
}

def get_processing_device(choice: DeviceChoice) -> str:
    has_gpu = torch.cuda.is_available()
    if choice == DeviceChoice.GPU and not has_gpu:
        raise ValueError("GPU foi solicitada, mas não está disponível ou configurada (CUDA).")
    if choice == DeviceChoice.AUTO:
        return 'cuda' if has_gpu else 'cpu'
    return choice.value.lower()

def filter_models_by_device(device: str) -> dict:
    if device == 'cpu':
        return {id: conf for id, conf in AVAILABLE_MODELS.items() if not conf.get('req_gpu', False)}
    return AVAILABLE_MODELS