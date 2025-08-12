# setup_worker.py
import sys
import os
import traceback
from logging_config import setup_root_logging, get_logger
from config import AVAILABLE_MODELS
from engine import load_model

# optional: huggingface cache helper
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

def run_setup():
    setup_root_logging()
    logger = get_logger("setup_worker")

    logger.info("="*80)
    logger.info("INICIANDO SETUP: pré-baixando/verificando modelos...")
    logger.info("="*80)

    failures = {}

    for model_id, cfg in AVAILABLE_MODELS.items():
        logger.info(f"Verificando modelo '{model_id}' -> {cfg.get('model_name')} ({cfg.get('impl')})")
        try:
            # tentativa simples usando engine.load_model (força download via libs)
            try:
                load_model(model_id, cfg, device='cpu')
                logger.info(f"✅ Modelo '{model_id}' carregado (via engine).")
            except Exception as e:
                logger.warning(f"Falha ao carregar '{model_id}' via engine: {e}")
                # Se for HF model e huggingface_hub disponível, tentar snapshot_download
                if snapshot_download and cfg.get('impl') == 'hf_pipeline':
                    try:
                        token = os.environ.get('HUGGINGFACE_HUB_TOKEN', None)
                        logger.info(f"Tentando snapshot_download para '{cfg.get('model_name')}'...")
                        snapshot_download(repo_id=cfg.get('model_name'), repo_type='model', use_auth_token=token)
                        logger.info(f"✅ snapshot_download concluído para '{cfg.get('model_name')}'.")
                    except Exception as e2:
                        logger.warning(f"snapshot_download falhou para '{cfg.get('model_name')}': {e2}")
                        raise

                else:
                    raise

        except Exception as e_final:
            failures[model_id] = str(e_final)
            logger.error(f"❌ Erro no setup do modelo '{model_id}': {e_final}")
            if logger.isEnabledFor(10):  # DEBUG level
                logger.debug(traceback.format_exc())

    logger.info("="*80)
    if failures:
        logger.warning(f"Setup finalizado com falhas em {len(failures)} modelo(s). Veja detalhes nos logs.")
        for mid, err in failures.items():
            logger.warning(f" - {mid}: {err}")
        logger.info("Você pode tentar rodar novamente ou checar a conectividade / token HF.")
        sys.exit(1)
    else:
        logger.info("✅ SETUP CONCLUÍDO: todos os modelos verificados com sucesso.")
        logger.info("="*80)
        sys.exit(0)

if __name__ == "__main__":
    run_setup()
