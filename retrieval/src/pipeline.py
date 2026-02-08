from typing import List
import time
import numpy as np
from indexing.src.logger import logger
from . import mmr


class RetrievalPipeline:
    def __init__(
        self,
        retriever,
        reranker,
        embedder
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.embedder = embedder

    def _select_by_token_budget(
        self,
        texts: List[str],
        max_tokens: int = 1500
    ) -> List[str]:
        selected = []
        total_tokens = 0

        for text in texts:
            tokens = len(text) // 4
            if total_tokens + tokens > max_tokens:
                break
            selected.append(text)
            total_tokens += tokens

        return selected

    def run(
        self,
        query: str,
        retrieve_k: int = 40,
        final_k: int = 5
    ) -> List[str]:

        start_time = time.time()
        logger.info(f"[PIPELINE] query: {query}")

        # ---------------- Retrieval ----------------
        try:
            hits = self.retriever.retrieve(
                query=query,
                top_k=retrieve_k
            )
        except Exception as e:
            logger.error(f"[PIPELINE] retrieval failed: {e}")
            return []

        passages = [
            h.payload.get("text", "")
            for h in hits
            if "text" in h.payload and len(h.payload["text"]) > 50
        ]

        logger.info(f"[PIPELINE] {len(passages)} passages after filtering")

        # ---------------- MMR ----------------
        doc_embeddings = np.array([
            self.embedder.embed([p], is_query=False)[0]
            for p in passages
        ])

        query_embedding = self.embedder.embed(
            [query],
            is_query=True
        )[0]

        selected_indices = mmr(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            k=min(10, len(passages))
        )

        passages = [passages[i] for i in selected_indices]

        logger.info(f"[PIPELINE] {len(passages)} passages after MMR")

        # ---------------- Re-ranking ----------------
        scores = self.reranker.rerank(
            query=query,
            passages=passages
        )

        ranked = sorted(
            zip(passages, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top_texts = [p for p, _ in ranked]

        # ---------------- Token budget ----------------
        final_contexts = self._select_by_token_budget(
            top_texts,
            max_tokens=1500
        )

        elapsed = time.time() - start_time
        logger.info(f"[PIPELINE] done in {elapsed:.2f}s")

        return final_contexts
