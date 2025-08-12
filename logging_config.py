import logging
import sys
import json
from config import DEBUG


def setup_root_logging():
    level = logging.DEBUG if DEBUG else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if root.hasHandlers():
        root.handlers.clear()

    fmt = "%(asctime)s - [%(levelname)s] - (%(name)s) - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root.addHandler(handler)

    # Quieter libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        rec = {"level": record.levelname, "name": record.name, "message": record.getMessage()}
        if record.exc_info:
            rec["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(rec)


def setup_worker_logging_json():
    """Configure logging in worker processes to emit JSON to stdout so manager can parse if needed."""
    level = logging.DEBUG if DEBUG else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    if root.hasHandlers():
        root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str):
    return logging.getLogger(name)
