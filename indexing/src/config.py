import os
from pathlib import Path

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    
    # --- EXPERIMENT SETTINGS ---
    
    # === EXPERIMENT 1: E5-Base ===
    # MODEL_NAME = "intfloat/multilingual-e5-base"
    # IS_E5_MODEL = True
    
    # === EXPERIMENT 2: E5-large ===
    MODEL_NAME = "intfloat/multilingual-e5-large"
    IS_E5_MODEL = True

    # === EXPERIMENT 3: GEMMA
    # MODEL_NAME = "google/gemma-embed" # Or "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # IS_E5_MODEL = False

    # --- AUTOMATIC CONFIGURATION ---
    
    # Clean up the model name to use as a folder name (e.g., "multilingual-e5-large")
    _clean_name = MODEL_NAME.split("/")[-1]
    
    # Dynamic Collection Name
    COLLECTION_NAME = f"legal_rag_{_clean_name}"
    
    # Dynamic Database Folder Path
    # This will create: "qdrant_db_multilingual-e5-large" or "qdrant_db_gemma-embed"
    QDRANT_PATH = BASE_DIR / f"qdrant_db_{_clean_name}"

    # --- Chunking & Performance ---
    MAX_TOKENS = 512
    SEMANTIC_THRESHOLD = 0.65 
    BATCH_SIZE = 64

    # --- INDEX SETTINGS ---
    # HNSW is the standard algorithm for Qdrant.
    # M: Number of edges per node in the graph (Higher = more memory, better accuracy).
    # ef_construct: Number of neighbours to consider during indexing (Higher = slower build, better accuracy).
    HNSW_M = 16 
    HNSW_EF = 100