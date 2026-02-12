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

    def _select_by_token_budget(self, candidates: List[tuple], max_tokens: int = 2000) -> List[Dict[str, Any]]:
        """
        candidates: List of (hit_object, score)
        Returns: List of formatted dictionaries
        """
        selected_results = []
        current_tokens = 0

        for hit, score in candidates:
            text = hit.payload.get("text", "")
            
            # Count tokens
            count = len(self.tokenizer.tokenize(text))
            
            if current_tokens + count > max_tokens:
                break
            
            # Construct the object with metadata
            result_obj = {
                "text": text,
                "metadata": hit.payload.get("metadata", {}),
                "doc_id": hit.payload.get("doc_id", "unknown"),
                "chunk_id": hit.id, # The unique Qdrant ID
                "score": score
            }
            
            selected_results.append(result_obj)
            current_tokens += count

        return selected_results

    def run(self, query: str, retrieve_k: int = 40) -> List[Dict[str, Any]]:
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
        
        if not valid_hits:
            return []

        # MMR
        # Extract vectors directly from hits if available
        if hasattr(valid_hits[0], 'vector') and valid_hits[0].vector is not None:
             doc_embeddings = np.array([h.vector for h in valid_hits])
        else:
             # Fallback: re-embed if vectors are missing
             texts = [h.payload["text"] for h in valid_hits]
             doc_embeddings = self.embedder.embed(texts)

        selected_indices = mmr.mmr(
            query_embedding=query_vector,
            doc_embeddings=doc_embeddings,
            lambda_=0.7, 
            k=min(20, len(valid_hits))
        )

        mmr_hits = [valid_hits[i] for i in selected_indices]
        passages = [h.payload["text"] for h in mmr_hits]

        logger.info(f"[PIPELINE] {len(passages)} passages selected by MMR")

        # Re-ranking
        scores = self.reranker.rerank(query=query, passages=passages)

        # Zip the ACTUAL OBJECTS (mmr_hits) with scores
        ranked_candidates = sorted(zip(mmr_hits, scores), key=lambda x: x[1], reverse=True)

        # Token Budget Selection
        final_results = self._select_by_token_budget(ranked_candidates, max_tokens=2500)

        elapsed = time.time() - start_time
        logger.info(f"[PIPELINE] done in {elapsed:.2f}s. Final count: {len(final_results)}")

        return final_results