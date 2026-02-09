import sys
import os
from pathlib import Path

# Fix python path to allow imports from sibling directories
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from indexing.src.embedding import Embedder
from indexing.src.config import Config
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker
from retrieval.src.pipeline import RetrievalPipeline
from retrieval.src.logger import setup_retrieval_logger

def main():
    logger = setup_retrieval_logger(Config.MODEL_NAME)

    # Load Embedder
    logger.info("⏳ Loading Embedder...")
    embedder = Embedder(
        model_name=Config.MODEL_NAME,
        is_e5=Config.IS_E5_MODEL
    )

    # Setup Retriever
    logger.info(f"📂 Opening Database: {Config.QDRANT_PATH}")
    retriever = Retriever(
        qdrant_path=str(Config.QDRANT_PATH), 
        collection_name=Config.COLLECTION_NAME,
        embedder=embedder
    )

    # Setup Re-ranker
    logger.info("⚖️  Loading Re-ranker...")
    reranker = ReRanker(
        model_name="BAAI/bge-reranker-v2-m3"
    )

    # Setup Pipeline
    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
        embedder=embedder
    )

    # Run Test
    query = "شرایط طلاق به درخواست زوج چیست؟"
    
    logger.info(f"\n🔍 Searching for: {query}...")
    results = pipeline.run(query=query, retrieve_k=50)

    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS ({len(results)} chunks selected)")
    logger.info("=" * 60)

    for i, text in enumerate(results, 1):
        logger.info(f"\n📄 [RESULT {i}]")
        logger.info("-" * 30)
        logger.info(text)
        logger.info("-" * 30)

if __name__ == "__main__":
    main()