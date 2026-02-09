from typing import List, Dict, Any

from retrieval.src.pipeline import RetrievalPipeline
from retrieval.src.context_builder import ContextBuilder
from .prompt import PromptBuilder
from .llm_client import LLMClient


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    Query → Retrieval → Context → Prompt → LLM → Answer
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: LLMClient,
        max_docs: int = 8,
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.context_builder = ContextBuilder(max_docs=max_docs)
        self.prompt_builder = PromptBuilder()
        self.llm_client = llm_client

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute full RAG pipeline.

        Returns:
            {
                "answer": str,
                "documents": dict,
                "used_docs": list[str]
            }
        """

        # 1️⃣ Retrieve + Re-rank
        hits = self.retrieval_pipeline.run(
            query=query,
            retrieve_k=50
        )

        # 2️⃣ Build context + document map
        context_str, doc_map = self.context_builder.build(hits)

        # 3️⃣ Build prompt
        messages = self.prompt_builder.build(
            query=query,
            context=context_str
        )

        # 4️⃣ Call LLM
        answer = self.llm_client.generate(
            system_prompt=messages["system"],
            user_prompt=messages["user"]
        )

        # 5️⃣ Extract used citations
        used_docs = [
            doc_id for doc_id in doc_map.keys()
            if doc_id in answer
        ]

        return {
            "answer": answer,
            "documents": doc_map,
            "used_docs": used_docs,
        }
