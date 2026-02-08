from indexing.src.embedding import Embedder
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker
from retrieval.src.pipeline import RetrievalPipeline


def main():
    # ---------------- Embedding Model ----------------
    embedder = Embedder(
        model_name="intfloat/multilingual-e5-base",
        is_e5=True
    )

    # ---------------- Retriever ----------------
    retriever = Retriever(
        qdrant_path="judicial-rag-system/qdrant_db_multilingual-e5-base",
        collection_name="legal_rag_multilingual-e5-base",
        embedder=embedder
    )

    # ---------------- Re-ranker ----------------
    reranker = ReRanker(
        model_name="BAAI/bge-reranker-v2-m3"
    )

    # ---------------- Pipeline ----------------
    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
        embedder=embedder
    )

    # ---------------- Test Query ----------------
    query = "شرایط قانونی فسخ قرارداد اجاره در ایران چیست؟"

    results = pipeline.run(
        query=query,
        retrieve_k=40,
        final_k=5
    )

    # ---------------- Output ----------------
    print("\n" + "=" * 80)
    print("QUERY:")
    print(query)
    print("=" * 80)

    for i, text in enumerate(results, 1):
        print(f"\n[CONTEXT {i}]")
        print("-" * 80)
        print(text[:800])  
        print("-" * 80)


if __name__ == "__main__":
    main()