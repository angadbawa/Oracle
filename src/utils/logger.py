"""Logging utilities for Oracle project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import get_config

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
        format_string: Custom log format string
        
    Returns:
        Configured logger instance
    """
    config = get_config()
    
    # Get configuration values
    if level is None:
        level = config.get('logging.level', 'INFO')
    
    if log_file is None:
        log_file = config.get('logging.file', './logs/oracle.log')
    
    if format_string is None:
        format_string = config.get(
            'logging.format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger for Oracle
    logger = logging.getLogger('oracle')
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")
    
    return logger

def get_logger(name: str = 'oracle') -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
