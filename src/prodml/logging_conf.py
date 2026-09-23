"""
JSON Structured Logging Configuration for prodml.
Ensures zero print() statements and outputs structured JSON logs.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom JSON Formatter ensuring standard structured output fields:
    timestamp, level, logger, message, correlation_id.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "N/A"),
        }

        # Include extra attributes if passed
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_object.update(record.extra)

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object)


def setup_logger(name: str = "prodml", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger with JSON formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if setup_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    logger.propagate = False
    return logger


# Default logger instance
logger = setup_logger()
