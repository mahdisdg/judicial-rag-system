from typing import List, Dict, Tuple
from qdrant_client.models import ScoredPoint


class ContextBuilder:
    def __init__(self, max_docs: int = 8):
        """
        max_docs: maximum number of documents to pass to the LLM
        """
        self.max_docs = max_docs

    def build(
        self,
        hits: List[ScoredPoint]
    ) -> Tuple[str, Dict[str, dict]]:
        """
        Build context string with citations and document map.

        Returns:
            context_str: formatted context for LLM
            doc_map: mapping DOC_ID -> metadata
        """

        context_blocks = []
        doc_map = {}

        for idx, hit in enumerate(hits[: self.max_docs], start=1):
            doc_id = f"DOC_{idx}"

            payload = hit.payload or {}
            text = payload.get("text", "").strip()

            if not text:
                continue

            context_blocks.append(
                f"[{doc_id}]\n{text}"
            )

            doc_map[doc_id] = {
                "point_id": str(hit.id),
                "score": float(hit.score),
                "metadata": payload,
            }

        context_str = "\n\n".join(context_blocks)

        return context_str, doc_map
