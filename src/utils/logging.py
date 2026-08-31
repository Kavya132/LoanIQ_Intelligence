"""
Logging utilities for the Loan Intelligence Engine
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    app_name: str = "loan_intelligence"
) -> None:
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        app_name: Application name prefix
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Format string
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main app log
        app_log_path = log_dir / f"{app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Training-specific log
        train_log_path = log_dir / "training.log"
        train_handler = logging.handlers.RotatingFileHandler(
            train_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        train_handler.setFormatter(formatter)
        train_logger = logging.getLogger("train")
        train_logger.addHandler(train_handler)
        
        # LLM-specific log
        llm_log_path = log_dir / "llm.log"
        llm_handler = logging.handlers.RotatingFileHandler(
            llm_log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        llm_handler.setFormatter(formatter)
        llm_logger = logging.getLogger("llm")
        llm_logger.addHandler(llm_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
