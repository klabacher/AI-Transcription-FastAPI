# Transcription API — faster-whisper + distil-ptbr

API de transcrição focalizada em:
- **faster-whisper** — modelos locais otimizados para inference rápida (GPU recomendado)
- **freds0/distil-whisper-large-v3-ptbr** — versão distil com suporte PT-BR via Hugging Face

## Instruções rápidas
1. Crie um ambiente virtual Python 3.10+
2. `pip install -r requirements.txt`
3. Inicie: `uvicorn main:app --reload`
4. Acesse `http://127.0.0.1:8000/docs` para testar endpoints

## Endpoints principais
- **POST /jobs** — cria jobs (form data: `model_id`, `session_id`, `language`, `files[]`, `device_choice`)
- **GET /jobs/{job_id}** — status do job
- **GET /jobs/{job_id}/download** — baixa transcrição
- **GET /models** — lista modelos disponíveis

## Observações
- Modelos grandes (distil-ptbr) são pesados — GPU reduces latency drastically.
- O design usa processos isolados (`multiprocessing`) para evitar pollução de memória ou conflitos entre ML libs e o servidor FastAPI.
