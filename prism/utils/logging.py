"""
PRISM Logging Configuration

Centralized logging setup for all PRISM modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging for PRISM.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Get root logger for prism
    logger = logging.getLogger("prism")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at {logging.getLevelName(level)} level")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a PRISM module.
    
    Args:
        name: Module name (will be prefixed with 'prism.')
        
    Returns:
        Logger instance
    """
    if not name.startswith("prism"):
        name = f"prism.{name}"
    return logging.getLogger(name)
