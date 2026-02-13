from typing import List, Dict, Any
import time
import numpy as np
from . import mmr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from rag_llm.src.logger import rag_logger

class RetrievalPipeline:
    def __init__(self, retriever, reranker, embedder):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder
        self.tokenizer = embedder.tokenizer

    def _select_by_token_budget(self, candidates: List[tuple], max_tokens: int = 2000) -> List[Dict[str, Any]]:
        selected_results = []
        current_tokens = 0
        for hit, score in candidates:
            text = hit.payload.get("text", "")
            count = len(self.tokenizer.tokenize(text))
            if current_tokens + count > max_tokens: break
            
            result_obj = {
                "text": text,
                "metadata": hit.payload.get("metadata", {}),
                "doc_id": hit.payload.get("doc_id", "unknown"),
                "chunk_id": hit.id,
                "score": score
            }
            selected_results.append(result_obj)
            current_tokens += count
        return selected_results

    def run(self, query: str, retrieve_k: int = 40) -> List[Dict[str, Any]]:
        start_time = time.time()
        rag_logger.info(f"🔎 [Retrieval] Searching for: '{query}'")

        # Retrieval
        try:
            hits, query_vector = self.retriever.retrieve(query=query, top_k=retrieve_k)
        except Exception as e:
            rag_logger.error(f"❌ [Retrieval] Failed: {e}")
            return []

        # Filter
        valid_hits = [h for h in hits if h.payload.get("text") and len(h.payload["text"]) > 20]
        rag_logger.debug(f"📊 [Retrieval] Hits found: {len(valid_hits)} (Filtered)")

        if not valid_hits:
            rag_logger.warning("⚠️ [Retrieval] No valid documents found.")
            return []

        # MMR
        if hasattr(valid_hits[0], 'vector') and valid_hits[0].vector is not None:
             doc_embeddings = np.array([h.vector for h in valid_hits])
        else:
             texts = [h.payload["text"] for h in valid_hits]
             doc_embeddings = self.embedder.embed(texts)

        selected_indices = mmr.mmr(
            query_embedding=query_vector,
            doc_embeddings=doc_embeddings,
            lambda_=0.7, 
            k=min(20, len(valid_hits))
        )
        mmr_hits = [valid_hits[i] for i in selected_indices]

        # Re-ranking
        passages = [h.payload["text"] for h in mmr_hits]
        scores = self.reranker.rerank(query=query, passages=passages)
        ranked_candidates = sorted(zip(mmr_hits, scores), key=lambda x: x[1], reverse=True)

        # Selection
        final_results = self._select_by_token_budget(ranked_candidates, max_tokens=2500)

        elapsed = time.time() - start_time
        
        # --- LOGGING ---
        rag_logger.info(f"✅ [Retrieval] Selected {len(final_results)} documents in {elapsed:.2f}s")
        for i, res in enumerate(final_results[:3]): # Log top 3 docs title
            title = res['metadata'].get('title', 'No Title')
            rag_logger.debug(f"   📄 Doc {i+1}: {title} (Score: {res['score']:.4f})")
        # ------------------------

        return final_results