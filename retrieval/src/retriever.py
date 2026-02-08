from qdrant_client import QdrantClient
from typing import List
from indexing.src.logger import logger


class Retriever:
    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embedder
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 40
    ):
        logger.info("[RETRIEVER] embedding query")

        query_vector = self.embedder.embed(
            [query],
            is_query=True
        )[0]

        logger.info(f"[RETRIEVER] searching top_k={top_k}")

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        logger.info(f"[RETRIEVER] retrieved {len(hits)} hits")
        return hits
