import logging
import sys
import json
import os

# Determine log level from environment variable or default to INFO
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


class JsonFormatter(logging.Formatter):
    """
    A custom formatter to output log records as a JSON string.
    This is ideal for consumption by log management systems like Loki.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_record)


def setup_root_logging():
    """
    Configures the root logger for the application.

    This setup uses the JsonFormatter to ensure all logs are structured,
    which is beneficial for a containerized environment.
    """
    level = logging.getLevelName(LOG_LEVEL)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers to avoid duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a handler that streams to stdout
    handler = logging.StreamHandler(sys.stdout)

    # Set the JSON formatter
    formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S%z")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Set higher log levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def setup_worker_logging_json():
    """
    This function is an alias for the main setup function.
    In V3, both the API and the worker use the same structured logging setup.
    """
    setup_root_logging()


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.
    """
    return logging.getLogger(name)
