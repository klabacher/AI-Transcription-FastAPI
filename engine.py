import logging
from typing import Generator, Dict, Any
from config import AVAILABLE_MODELS

logger = logging.getLogger('engine')
MODEL_INSTANCES: Dict[str, Any] = {}


def load_model(model_id: str, config: dict, device: str):
    key = f"{model_id}:{device}"
    if key in MODEL_INSTANCES:
        return MODEL_INSTANCES[key]

    impl = config.get('impl')
    name = config.get('model_name')
    logger.info(f"Carregando modelo {model_id} ({impl}) para dispositivo {device}")

    if impl == 'faster':
        from faster_whisper import WhisperModel
        compute_type = config.get('compute_type', None)
        model = WhisperModel(name, device=device, compute_type=compute_type)

    elif impl == 'hf_pipeline':
        from transformers import pipeline
        torch_device = 0 if device == 'cuda' else -1
        # return_timestamps is used when calling
        model = pipeline('automatic-speech-recognition', model=name, device=torch_device)

    else:
        raise ValueError(f"Implementação desconhecida: {impl}")

    MODEL_INSTANCES[key] = model
    logger.info(f"Modelo {model_id} carregado com sucesso")
    return model


def transcribe_with_faster(model, audio_path: str, language: str, duration: float) -> Generator:
    """Gera progresso (int) e items dict com segmento ou resultado final."""
    segments, info = model.transcribe(audio_path, language=language)
    full_parts = []
    # segments is iterable of objects with start, end, text
    for seg in segments:
        full_parts.append(seg.text)
        # progress estimate based on segment end and duration
        if duration and duration > 0:
            progress = int((seg.end / duration) * 100)
            progress = max(0, min(99, progress))
            yield progress
        else:
            yield 50
        yield {'segment': {'start': seg.start, 'end': seg.end, 'text': seg.text}}

    yield {'text': ''.join(full_parts).strip(), 'segments': [{'start': s.start, 'text': s.text} for s in segments]}


def transcribe_with_hf_pipeline(model, audio_path: str, language: str) -> Generator:
    # HuggingFace pipeline often returns full result; emulate progress
    yield 10
    try:
        result = model(audio_path, return_timestamps=True)
    except TypeError:
        # Some pipelines expect a dict param
        result = model(audio_path)

    yield 80

    text = ''
    chunks = []
    if isinstance(result, dict):
        text = result.get('text', '')
        chunks = result.get('chunks', []) or result.get('segments', [])
    elif isinstance(result, str):
        text = result
    # normalize chunks
    segments = []
    for c in chunks:
        # chunk may have 'timestamp' or 'start'
        start = None
        if isinstance(c.get('timestamp', None), (list, tuple)):
            start = c['timestamp'][0]
        elif 'start' in c:
            start = c['start']
        segments.append({'start': start or 0, 'text': c.get('text', '')})

    yield {'text': text.strip(), 'segments': segments}
