import logging
import sys
import json
from config import DEBUG

class JsonFormatter(logging.Formatter):
    """Formata logs como uma string JSON para comunicação entre processos."""
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_record)

def setup_logging():
    """Configura o logging para a aplicação principal."""
    level = logging.DEBUG if DEBUG else logging.INFO
    
    # Formato para o console principal
    console_formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Handler para o console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Configura o logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Limpa handlers existentes para evitar duplicação
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(console_handler)
    
    # Silencia loggers muito verbosos de bibliotecas de terceiros
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

def setup_worker_logging():
    """Configura o logging para os processos worker, redirecionando para stderr."""
    level = logging.DEBUG if DEBUG else logging.INFO
    
    # Formato JSON para o worker
    json_formatter = JsonFormatter()
    
    # Handler que escreve no stderr (nosso "intercomunicador")
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(json_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(stderr_handler)

def get_logger(name):
    """Função de ajuda para obter um logger com o nome do módulo."""
    return logging.getLogger(name)