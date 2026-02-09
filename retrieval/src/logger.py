import logging
import os
from datetime import datetime
from pathlib import Path
from indexing.src.config import Config

def setup_retrieval_logger(experiment_name: str):
    """
    Sets up a logger for the retrieval pipeline that writes to the central logs directory.
    """
    # Use the BASE_DIR from indexing config to find the logs folder
    log_dir = Config.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create Unique Filename starting with 'retrieval_'
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_exp_name = experiment_name.split("/")[-1]
    log_file = log_dir / f"retrieval_{clean_exp_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("RetrievalLogger")
    logger.info(f"🚀 Retrieval Log file created at: {log_file}")
    return logger

# Generic logger instance for use inside classes
logger = logging.getLogger("RetrievalLogger")