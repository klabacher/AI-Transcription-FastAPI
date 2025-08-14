import sys
import os
import traceback

# Garante que o script encontre os outros módulos
sys.path.append(os.getcwd())

from config import AVAILABLE_MODELS
from logging_config import setup_root_logging, get_logger

# Importa as bibliotecas necessárias para o download
try:
    from faster_whisper import WhisperModel
    from transformers import pipeline as hf_pipeline
    from huggingface_hub import snapshot_download
except ImportError as e:
    print(f"ERRO: Biblioteca de IA não encontrada: {e}", file=sys.stderr)
    print(
        "Por favor, instale as dependências com 'pip install -r requirements.txt'",
        file=sys.stderr,
    )
    sys.exit(1)


def run_setup():
    """
    Script de setup para baixar e cachear todos os modelos necessários
    antes da primeira execução da API.
    """
    setup_root_logging()
    logger = get_logger("setup_worker")

    logger.info("=" * 80)
    logger.info("INICIANDO VERIFICAÇÃO E DOWNLOAD DOS MODELOS...")
    logger.info("Isso pode demorar MUITO tempo e consumir bastante espaço em disco.")
    logger.info("As próximas inicializações da API serão rápidas.")
    logger.info("=" * 80)

    success_count = 0
    failure_count = 0

    for model_id, config in AVAILABLE_MODELS.items():
        logger.info(f"--- Processando modelo: '{model_id}' ---")
        model_name = config["model_name"]
        impl = config["impl"]

        try:
            if impl == "faster":
                logger.info(f"Baixando '{model_name}' para faster-whisper...")
                # Baixa o modelo usando a própria biblioteca, que gerencia o cache
                _ = WhisperModel(model_name, device="cpu", compute_type="int8")

            elif impl == "hf_pipeline":
                logger.info(f"Baixando '{model_name}' do Hugging Face Hub...")
                # Usa snapshot_download para baixar todos os arquivos do repositório
                snapshot_download(
                    repo_id=model_name,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "*.md",
                        "preprocessor_config.json",
                    ],
                )

            logger.info(f"✅ Modelo '{model_id}' baixado e verificado com sucesso.")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Falha no setup do modelo '{model_id}' ({model_name}).")
            logger.error(f"   Erro: {e}")
            logger.debug(traceback.format_exc())
            failure_count += 1
        logger.info("-" * (len(model_id) + 24))

    logger.info("\n" + "=" * 80)
    if failure_count > 0:
        logger.warning(f"SETUP CONCLUÍDO COM {failure_count} FALHA(S).")
        logger.warning(
            "A API pode não funcionar corretamente com os modelos que falharam."
        )
        sys.exit(1)
    else:
        logger.info(
            f"✅ SETUP CONCLUÍDO COM SUCESSO! ({success_count} modelos prontos)"
        )
        logger.info("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    run_setup()
