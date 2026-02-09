from typing import List, Dict, Any
import time
import numpy as np
from .logger import logger
from . import mmr

class RetrievalPipeline:
    def __init__(self, retriever, reranker, embedder):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder
        self.tokenizer = embedder.tokenizer

    def _select_by_token_budget(self, texts: List[str], max_tokens: int = 1500) -> List[str]:
        selected = []
        current_tokens = 0

        for text in texts:
            # Token counting
            count = len(self.tokenizer.tokenize(text))
            
            if current_tokens + count > max_tokens:
                break
            
            selected.append(text)
            current_tokens += count

        return selected

    def run(self, query: str, retrieve_k: int = 40, final_k: int = 5) -> List[str]:
        start_time = time.time()
        logger.info(f"[PIPELINE] query: {query}")

        # Retrieval
        try:
            hits, query_vector = self.retriever.retrieve(query=query, top_k=retrieve_k)
        except Exception as e:
            logger.error(f"[PIPELINE] retrieval failed: {e}")
            return []

        # Filter empty texts
        valid_hits = [h for h in hits if h.payload.get("text") and len(h.payload["text"]) > 20]
        logger.info(f"[PIPELINE] {len(valid_hits)} hits after filtering")

        if not valid_hits:
            return []

        # MMR
        # Extract vectors directly from Qdrant hits
        doc_embeddings = np.array([h.vector for h in valid_hits])
        
        # Run MMR
        selected_indices = mmr.mmr(
            query_embedding=query_vector,
            doc_embeddings=doc_embeddings,
            lambda_=0.7, # 0.7 = Balance between relevance and diversity
            k=min(15, len(valid_hits)) # Select slightly more for re-ranking
        )

        # Get the actual objects selected by MMR
        mmr_hits = [valid_hits[i] for i in selected_indices]
        passages = [h.payload["text"] for h in mmr_hits]

        logger.info(f"[PIPELINE] {len(passages)} passages selected by MMR")

        # Re-ranking
        # We re-rank the diverse set found by MMR
        scores = self.reranker.rerank(query=query, passages=passages)

        # Sort by Re-ranker score
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        top_texts = [p for p, _ in ranked]

        # Token Budget Selection
        final_contexts = self._select_by_token_budget(top_texts, max_tokens=2000)

        elapsed = time.time() - start_time
        logger.info(f"[PIPELINE] done in {elapsed:.2f}s. Final count: {len(final_contexts)}")

        return final_contexts