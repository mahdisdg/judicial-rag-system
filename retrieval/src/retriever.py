from qdrant_client import QdrantClient
from typing import List, Any
from .logger import logger

class Retriever:
    def __init__(
        self,
        qdrant_path: str,
        collection_name: str,
        embedder
    ):
        logger.info(f"[RETRIEVER] Connecting to Qdrant at: {qdrant_path}")
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 40):
            logger.info("[RETRIEVER] embedding query")
            
            # Embed the query
            query_vector = self.embedder.embed([query], is_query=True)[0]

            logger.info(f"[RETRIEVER] searching top_k={top_k}")

            # Search Qdrant
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                with_vectors=True 
            )
            
            # Extract the list
            hits = response.points 

            logger.info(f"[RETRIEVER] retrieved {len(hits)} hits")
            return hits, query_vector