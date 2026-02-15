import logging
import os
from datetime import datetime
from pathlib import Path
from config.config import Config

def setup_logger(experiment_name: str):
    """
    Sets up a logger that writes to both File and Console.
    """
    # Create logs directory
    log_dir = Config.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create Unique Filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_exp_name = experiment_name.split("/")[-1]
    log_file = log_dir / f"indexing_{clean_exp_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("LegalRAG")
    logger.info(f"📝 Log file created at: {log_file}")
    return logger

# Generic logger instance
logger = logging.getLogger("LegalRAG")