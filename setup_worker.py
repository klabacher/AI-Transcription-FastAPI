import sys
import os

# Garante que o script possa encontrar os outros módulos do projeto
sys.path.append(os.getcwd())

from config import AVAILABLE_MODELS
from logging_config import setup_logging, get_logger

def run_setup():
    """
    Este script é o "peão de setup". Sua única função é garantir que todos os
    modelos locais sejam baixados e cacheados pelo menos uma vez.
    """
    setup_logging()
    logger = get_logger("setup_worker")

    logger.info("="*80)
    logger.info("INICIANDO VERIFICAÇÃO E DOWNLOAD DOS MODELOS (PODE DEMORAR BASTANTE)...")
    logger.info("Isso só acontecerá uma vez. As próximas inicializações serão instantâneas.")
    logger.info("="*80)

    try:
        from faster_whisper import WhisperModel
        from transformers import pipeline as hf_pipeline
        import whisper as openai_whisper

        models_to_setup = {k: v for k, v in AVAILABLE_MODELS.items() if v['impl'] != 'assemblyai'}

        for model_id, config in models_to_setup.items():
            try:
                logger.info(f"Verificando modelo: '{model_id}'...")
                
                if config['impl'] == 'openai':
                    openai_whisper.load_model(config['model_name'], device='cpu')
                elif config['impl'] == 'faster':
                    WhisperModel(config['model_name'], device='cpu', compute_type='int8')
                elif config['impl'] == 'hf_pipeline':
                    hf_pipeline("automatic-speech-recognition", model=config['model_name'])

                logger.info(f"✅ Modelo '{model_id}' está pronto para uso.")
            except Exception as e:
                logger.warning(f"Falha ao baixar ou verificar o modelo '{model_id}'.")
                logger.warning(f"   Erro: {e}")

        logger.info("\n" + "="*80)
        logger.info("✅ SETUP DE MODELOS CONCLUÍDO COM SUCESSO!")
        logger.info("="*80)
        
    except ImportError as e:
        logger.critical(f"Bibliotecas de IA não encontradas. {e}")
        logger.critical("Por favor, rode 'pip install -r requirements.txt' e tente novamente.")
        sys.exit(1)
    except Exception as e:
        import traceback
        logger.critical(f"ERRO INESPERADO DURANTE O SETUP: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    run_setup()