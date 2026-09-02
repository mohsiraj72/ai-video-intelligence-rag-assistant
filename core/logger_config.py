"""
Logging configuration for AI Video Assistance.
Centralized logging for better debugging and monitoring.
"""

import logging
import os
from datetime import datetime

# Create logs directory
os.makedirs("logs", exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Create logger
logger = logging.getLogger("ai_video_assistance")
logger.setLevel(getattr(logging, LOG_LEVEL))

# File handler
log_file = f"logs/ai_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_LEVEL))

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"ai_video_assistance.{name}")
