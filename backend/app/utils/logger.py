"""Centralized logging utility for VBC AI application."""

import logging
import sys
from datetime import datetime
from pathlib import Path


class VBCLogger:
    """Centralized logger utility for VBC AI application."""

    _loggers = {}
    _configured = False

    @classmethod
    def setup_logging(cls) -> None:
        """Setup global logging configuration."""
        if cls._configured:
            return

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        simple_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)

        # File handler for all logs
        file_handler = logging.FileHandler(
            log_dir / f"vbc_ai_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

        # Error file handler for errors only
        error_handler = logging.FileHandler(
            log_dir / f"vbc_ai_errors_{datetime.now().strftime('%Y%m%d')}.log",
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)

        # Set third-party library log levels
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)

        cls._configured = True

        # Log the setup completion
        logger = cls.get_logger("logger_setup")
        logger.info("VBC AI logging system initialized successfully")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance with the given name."""
        if not cls._configured:
            cls.setup_logging()

        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(f"vbc_ai.{name}")

        return cls._loggers[name]

    @classmethod
    def get_module_logger(cls, module_name: str) -> logging.Logger:
        """Get a logger for a specific module (e.g., __name__)."""
        # Extract the module name from full path
        if module_name.startswith("backend.app."):
            clean_name = module_name.replace("backend.app.", "")
        else:
            clean_name = module_name.split(".")[-1]

        return cls.get_logger(clean_name)


# Convenience functions for easy import
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return VBCLogger.get_logger(name)


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module (use with __name__)."""
    return VBCLogger.get_module_logger(module_name)


def setup_logging() -> None:
    """Setup global logging configuration."""
    VBCLogger.setup_logging()


# Auto-setup logging when module is imported
setup_logging()
