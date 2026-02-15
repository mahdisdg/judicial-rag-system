import os
from pathlib import Path

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    DB_ROOT_DIR = BASE_DIR / "DBs"
    DB_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # EXPERIMENT CONTROL PANEL
    # =========================================================================

    # --- SETTINGS FOR CURRENT RUN ---
    
    # Folder name for logs

    EXPERIMENT_NAME = "Exp8_ParsBERT_Rerank_qwen2.5-3b"
    # EXPERIMENT_NAME = "Exp2_E5_Base_Rerank_GPT4oMini"

    # Embedding Model

    EMBEDDING_MODEL = "HooshvareLab/bert-base-parsbert-uncased" # ParsBERT
    IS_E5_MODEL = False
    # EMBEDDING_MODEL = "intfloat/multilingual-e5-base"           # E5-Base
    # IS_E5_MODEL = True
    
    # Reranker
    USE_RERANKER = True   # True or False
    
    # LLM Model
    LLM_MODEL = "qwen2.5-vl-3b-instruct"  # Small (3B)
    # LLM_MODEL = "gpt-4o-mini"   # Medium (8B)
    
    # =========================================================================

    # --- AUTOMATIC PATH CONFIGURATION ---
    _clean_model_name = EMBEDDING_MODEL.split("/")[-1]
    
    # Collection & DB Path derived from the selected Embedding Model
    COLLECTION_NAME = f"legal_rag_{_clean_model_name}"
    QDRANT_PATH = DB_ROOT_DIR / f"qdrant_{_clean_model_name}"

    # Fixed Settings
    RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
    MAX_TOKENS = 512
    SEMANTIC_THRESHOLD = 0.65 
    BATCH_SIZE = 64
    HNSW_M = 16 
    HNSW_EF = 100