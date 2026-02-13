import os
from pathlib import Path

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    
    # Central DB Folder
    DB_ROOT_DIR = BASE_DIR / "DBs"
    DB_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- EXPERIMENT SETTINGS ---
    # Experiment 1: e5-base
    MODEL_NAME = "intfloat/multilingual-e5-base"
    IS_E5_MODEL = True
    
    # Experiment 2: heydariAI
    # MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"
    # IS_E5_MODEL = False

    # --- AUTOMATIC CONFIGURATION ---
    _clean_name = MODEL_NAME.split("/")[-1]
    
    COLLECTION_NAME = f"legal_rag_{_clean_name}"
    
    # Qdrant Path (Folder inside DBs)
    QDRANT_PATH = DB_ROOT_DIR / f"qdrant_{_clean_name}"

    # --- Chunking ---
    MAX_TOKENS = 512
    SEMANTIC_THRESHOLD = 0.65 
    BATCH_SIZE = 64
    
    # --- Qdrant HNSW Settings ---
    HNSW_M = 16 
    HNSW_EF = 100