from sentence_transformers import SentenceTransformer
from typing import List
import torch
import numpy as np
from .logger import logger

class Embedder:
    def __init__(self, model_name: str, is_e5: bool = False):
        logger.info(f"🔄 Loading Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Using Device: {self.device}")
        
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device, local_files_only=False)
        self.is_e5 = is_e5

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.array([])

        if self.is_e5:
            prefix = "query: " if is_query else "passage: "
            texts = [f"{prefix}{t}" for t in texts]
            
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embeddings