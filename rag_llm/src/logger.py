import logging
import os
from datetime import datetime
from pathlib import Path

# Define Project Root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_rag_logger():
    """
    Creates a logger that saves detailed execution traces.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"rag_trace_{timestamp}.log"

    logger = logging.getLogger("RAG_TRACE")
    logger.setLevel(logging.DEBUG) # Capture everything
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger

# Global instance
rag_logger = setup_rag_logger()