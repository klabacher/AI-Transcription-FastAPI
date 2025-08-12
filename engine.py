import torch
from faster_whisper import WhisperModel
from transformers import pipeline as hf_pipeline
from logging_config import get_logger

logger = get_logger("engine")

def load_model_for_worker(model_id: str, config: dict, device: str):
    impl = config['impl']
    model_name = config['model_name']
    
    logger.info(f"Carregando modelo '{model_id}' ({impl}) com nome '{model_name}' para o dispositivo '{device}'...")

    if impl == 'faster':
        compute_type = config.get('compute_type', 'default')
        if device == 'cpu' and compute_type not in ['int8', 'float32']:
            logger.warning(f"Compute type '{compute_type}' não é otimizado para CPU. Usando 'int8' por padrão.")
            compute_type = 'int8'
        elif device == 'cuda' and compute_type not in ['float16', 'int8_float16']:
             logger.warning(f"Compute type '{compute_type}' não é ideal para GPU. Considere 'float16' ou 'int8_float16'.")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    elif impl == 'hf_pipeline':
        torch_device_id = 0 if device == 'cuda' else -1
        model = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=torch_device_id,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
    else:
        raise ValueError(f"Implementação de modelo desconhecida: {impl}")
    
    logger.info(f"Modelo '{model_id}' carregado com sucesso no worker.")
    return model

def transcribe_audio(model, model_config: dict, audio_path: str, duration_seconds: float):
    impl = model_config['impl']
    language_code = 'pt'

    if impl == 'faster':
        segments, info = model.transcribe(audio_path, language=language_code, vad_filter=True)
        logger.debug(f"faster-whisper detectou idioma: {info.language} (prob: {info.language_probability:.2f})")
        
        all_segments = []
        full_text_parts = []
        for segment in segments:
            full_text_parts.append(segment.text)
            all_segments.append({'start': segment.start, 'text': segment.text.strip()})
            if duration_seconds > 0:
                progress = min(99, int((segment.end / duration_seconds) * 100))
                yield progress
        
        yield {"text": "".join(full_text_parts).strip(), "segments": all_segments}
    elif impl == 'hf_pipeline':
        yield 10
        kwargs = {
            "chunk_length_s": 30,
            "stride_length_s": 5,
            "return_timestamps": True,
            "generate_kwargs": {"language": "portuguese"}
        }
        result = model(audio_path, **kwargs)
        yield 80
        text_result = result.get('text', '').strip()
        segments_result = []
        if 'chunks' in result:
            segments_result = [{'start': c['timestamp'][0], 'text': c['text'].strip()} for c in result.get('chunks', [])]
        yield {"text": text_result, "segments": segments_result}