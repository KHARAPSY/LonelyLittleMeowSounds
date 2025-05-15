import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str, log_file: str = "lonelylittlemeowsounds.log", log_level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger with console and rotating file handlers.

    Args:
        name (str): Name of the logger (usually __name__ of the calling module).
        log_file (str): Path to the log file. Defaults to 'lonelylittlemeowsounds.log'.
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Defaults to 'INFO'.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        log_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=3
        )
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger