import os
from pathlib import Path

class Config:
    # --- Paths ---
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    # Local Qdrant
    QDRANT_PATH = BASE_DIR / "qdrant_local_db"

    # --- Model Settings ---
    # Using a high-quality model since you have Colab Pro
    MODEL_NAME = "intfloat/multilingual-e5-large"
    COLLECTION_NAME = "legal_rag_full_text"
    IS_E5_MODEL = True
    
    # EXPERIMENT B: Gemma (Uncomment to use)
    # MODEL_NAME = "google/gemma-embed"
    # COLLECTION_NAME = "legal_rag_gemma"
    # IS_E5_MODEL = False

    # --- Chunking Constraints ---
    MAX_TOKENS = 512       # Hard limit for the Embedding Model
    BATCH_SIZE = 64
    
    # --- Semantic Chunking Magic Numbers ---
    # Similarity Threshold (0.0 to 1.0). 
    # Sentences with similarity BELOW this will cause a split.
    # Higher = More chunks (More sensitive to topic changes).
    # Lower = Fewer, larger chunks.
    SEMANTIC_THRESHOLD = 0.65