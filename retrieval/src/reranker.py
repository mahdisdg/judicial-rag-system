from sentence_transformers import CrossEncoder
from typing import List
from indexing.src.logger import logger


def truncate(text: str, max_tokens: int = 512) -> str:
    return text[: max_tokens * 4]


class ReRanker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda"
    ):
        logger.info(f"[RERANKER] loading model: {model_name}")
        self.model = CrossEncoder(
            model_name,
            device=device
        )

    def rerank(
        self,
        query: str,
        passages: List[str],
        batch_size: int = 8
    ) -> List[float]:

        logger.info(f"[RERANKER] reranking {len(passages)} passages")

        passages = [
            truncate(p)
            for p in passages
        ]

        pairs = [(query, p) for p in passages]

        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )

        return scores
