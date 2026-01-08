from sentence_transformers import SentenceTransformer
from typing import List
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name: str, is_e5: bool = False):
        print(f"🔄 Loading Model: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Using Device: {self.device}")
        
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        self.is_e5 = is_e5

    @property
    def tokenizer(self):
        return self.model.tokenizer

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generates embeddings. 
        Auto-adds 'passage:' or 'query:' prefix for E5 models.
        """
        if not texts:
            return np.array([])

        if self.is_e5:
            prefix = "query: " if is_query else "passage: "
            # Avoid double prefixing if it's already there
            texts = [f"{prefix}{t}" if not t.startswith("passage:") and not t.startswith("query:") else t for t in texts]
            
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        return embeddings