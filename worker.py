import sys
import os
import io
import json
import soundfile as sf

# Adiciona o diretório atual ao path para que o worker possa importar os outros módulos
sys.path.append(os.getcwd())

# Importa a engine de forma segura, sabendo que o path está correto
from engine import run_local_transcription, run_assemblyai_transcription
from pathlib import Path
import tempfile

def worker_task(json_job_data, audio_bytes):
    """
    Esta é a função que roda no processo isolado (o "bunker").
    Agora ela se comunica via JSON.
    """
    try:
        # Desempacota os dados do job que o gerente (main.py) enviou
        job = json.loads(json_job_data)
        internal_path = job['internal_path']
        
        duration_seconds = 0
        try:
            audio_info = sf.info(io.BytesIO(audio_bytes))
            duration_seconds = audio_info.duration
        except Exception:
            pass

        # Cria um arquivo temporário para a engine usar
        suffix = Path(internal_path).suffix
        tmp_audio_path = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_audio_path = tmp.name

        impl = job['config']['model_config']['impl']
        if impl == 'assemblyai':
            result = run_assemblyai_transcription(job, tmp_audio_path)
        else:
            result = run_local_transcription(job, tmp_audio_path, duration_seconds)

        os.remove(tmp_audio_path)

        # Prepara a resposta de sucesso e a envia como uma string JSON
        success_response = json.dumps({"status": "completed", "result": result})
        sys.stdout.write(success_response)
        sys.stdout.flush()

    except Exception as e:
        import traceback
        error_info = {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        # Prepara a resposta de erro e a envia como uma string JSON
        error_response = json.dumps(error_info)
        sys.stderr.write(error_response)
        sys.stderr.flush()

if __name__ == "__main__":
    # O protocolo de comunicação agora é baseado em strings JSON com prefixo de tamanho
    job_data_size_bytes = sys.stdin.buffer.read(4)
    job_data_size = int.from_bytes(job_data_size_bytes, 'big')
    json_job_data = sys.stdin.buffer.read(job_data_size).decode('utf-8')

    audio_data_size_bytes = sys.stdin.buffer.read(4)
    audio_data_size = int.from_bytes(audio_data_size_bytes, 'big')
    audio_bytes = sys.stdin.buffer.read(audio_data_size)
    
    worker_task(json_job_data, audio_bytes)