import logging
import os
from datetime import datetime
from pathlib import Path
from .config import Config

def setup_logger(experiment_name: str):
    """
    Sets up a logger that writes to both File and Console.
    Filename format: logs/indexing_{model_name}_{timestamp}.log
    """
    
    # Create logs directory in the Project Root
    log_dir = Config.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create Unique Filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clean_exp_name = experiment_name.replace("/", "-") # Safety for file paths
    log_file = log_dir / f"indexing_{clean_exp_name}_{timestamp}.log"

    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            # Write to File
            logging.FileHandler(log_file, encoding='utf-8'),
            # Write to Console
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("LegalRAG")
    logger.info(f"📝 Log file created at: {log_file}")
    return logger

# Create a generic logger instance for modules to import
logger = logging.getLogger("LegalRAG")