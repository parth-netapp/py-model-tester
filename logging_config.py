import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOG_FILE_BASE = "py_model_tester"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger that logs to stdout and a file.

    The log file is created next to this module as `py_model_tester.log`.
    Configuration is applied only once per logger name.
    """
    logger_name = name or "py_model_tester"
    logger = logging.getLogger(logger_name)

    # Avoid reconfiguring if handlers are already attached
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = Path(__file__).resolve().parent
    
    # Create timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{_LOG_FILE_BASE}_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False

    return logger
