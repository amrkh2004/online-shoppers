import json
import logging

from prodml.logging_conf import JSONFormatter, setup_logger


def test_json_formatter_basic():
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test log message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Test log message"
    assert "timestamp" in data
    assert data["correlation_id"] == "N/A"


def test_json_formatter_extra_and_exception():
    formatter = JSONFormatter()
    try:
        raise ValueError("Sample error")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="test.py",
        lineno=20,
        msg="Error occurred",
        args=(),
        exc_info=exc_info,
    )
    record.extra = {"user_id": 123}

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["user_id"] == 123
    assert "exception" in data
    assert "Sample error" in data["exception"]


def test_setup_logger():
    logger_inst = setup_logger("test_prodml_custom", logging.DEBUG)
    assert logger_inst.name == "test_prodml_custom"
    assert logger_inst.level == logging.DEBUG
    # Call again to ensure handler non-duplication
    logger_inst2 = setup_logger("test_prodml_custom", logging.DEBUG)
    assert len(logger_inst2.handlers) == 1
