import torch
import tempfile
import os
import io
from pathlib import Path

try:
    import whisper as openai_whisper
    from faster_whisper import WhisperModel
    from transformers import pipeline as hf_pipeline
    import assemblyai as aai
except ImportError as e:
    raise ImportError(f"Biblioteca de ML faltando: {e}. Rode 'pip install -r requirements.txt'.")

MODEL_CACHE = {}

def get_model(model_id: str, config: dict, device: str):
    cache_key = f"{model_id}_{device}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    impl = config['impl']
    if impl == 'assemblyai':
        return None

    model_name = config['model_name']
    print(f"INFO: Carregando modelo '{model_id}' para dispositivo '{device}'...")

    if impl == 'openai':
        model = openai_whisper.load_model(model_name, device=device)
    elif impl == 'faster':
        compute_type = config.get('compute_type', 'default')
        if device == 'cpu' and compute_type not in ['int8', 'float32']:
            compute_type = 'int8'
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    elif impl == 'hf_pipeline':
        torch_device = 0 if device == 'cuda' else -1
        model = hf_pipeline("automatic-speech-recognition", model=model_name, device=torch_device)
    else:
        raise ValueError(f"Implementação desconhecida: {impl}")
    
    MODEL_CACHE[cache_key] = model
    return model

def run_transcription(job: dict, audio_bytes: bytes, internal_path: str, duration_seconds: float) -> dict:
    job_config = job['config']
    impl = job_config['model_config']['impl']
    
    suffix = Path(internal_path).suffix
    tmp_audio_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_audio_path = tmp.name

        if impl == 'assemblyai':
            # Para AssemblyAI, o cancelamento é antes do envio, pois não podemos matar o processo deles.
            if job.get('status') == 'cancelling': raise InterruptedError("Job cancelado antes do envio para a API.")
            return run_assemblyai_transcription(job, tmp_audio_path)
        else:
            return run_local_transcription(job, tmp_audio_path, duration_seconds)

    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

def run_assemblyai_transcription(job: dict, audio_path: str) -> dict:
    job_config = job['config']
    aai.settings.api_key = job_config['assemblyai_api_key']
    
    config = aai.TranscriptionConfig(
        speaker_labels=job_config.get('speaker_labels', False),
        entity_detection=job_config.get('entity_detection', False),
        language_code=job_config['language']
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Erro da API AssemblyAI: {transcript.error}")

    return {
        "text": transcript.text or " ",
        "segments": [vars(utt) for utt in transcript.utterances] if transcript.utterances else [],
        "entities": [vars(ent) for ent in transcript.entities] if transcript.entities else []
    }

def run_local_transcription(job: dict, audio_path: str, duration_seconds: float) -> dict:
    job_config = job['config']
    model_config = job_config['model_config']
    
    # A decisão de device agora é feita no worker, aqui apenas usamos
    from config import get_processing_device, DeviceChoice
    device_choice_str = job_config.get("device_choice", "AUTOMATICO")
    device_choice = DeviceChoice(device_choice_str)
    device = get_processing_device(device_choice)
    
    model = get_model(job_config['model_id'], model_config, device)
    
    text_result, segments_result = "", []
    language_code = job_config['language']
    
    if model_config['impl'] == 'faster':
        segments, _ = model.transcribe(audio_path, language=language_code)
        full_text_parts = []
        for segment in segments:
            # A checagem de cancelamento foi movida para o worker/main.py que pode matar o processo
            full_text_parts.append(segment.text)
            segments_result.append({'start': segment.start, 'text': segment.text.strip(), 'speaker': None})
        text_result = "".join(full_text_parts).strip()
    
    else:
        if model_config['impl'] == 'openai':
            result = model.transcribe(audio_path, language=language_code, fp16=(device == 'cuda'))
            text_result = result.get('text', ' ')
            if 'segments' in result:
                segments_result = [{'start': s['start'], 'text': s['text'], 'speaker': None} for s in result['segments']]

        elif model_config['impl'] == 'hf_pipeline':
            language_map = {'pt': 'portuguese', 'en': 'english'}
            task_language = language_map.get(language_code, 'portuguese')
            generate_kwargs = {"language": task_language, "task": "transcribe"}
            
            kwargs = {"return_timestamps": True, "generate_kwargs": generate_kwargs}
            if "distil-whisper" in model_config['model_name']:
                 kwargs.update({"chunk_length_s": 30, "stride_length_s": 5})

            result = model(audio_path, **kwargs)
            text_result = result.get('text', ' ').strip()
            if 'chunks' in result:
                segments_result = [{'start': c['timestamp'][0], 'text': c['text'].strip(), 'speaker': None} for c in result.get('chunks', [])]
        
    return {"text": text_result, "segments": segments_result, "entities": []}