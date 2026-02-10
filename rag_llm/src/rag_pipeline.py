from typing import Dict, Any, List
import logging

from retrieval.src.pipeline import RetrievalPipeline
from retrieval.src.context_builder import ContextBuilder
from .llm_client import LLMClient
from .prompt import PromptBuilder

logger = logging.getLogger("LegalRAG")

class RAGPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: LLMClient,
        max_docs_in_context: int = 8
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client
        
        # Helper classes
        self.context_builder = ContextBuilder(max_docs=max_docs_in_context)
        self.prompt_builder = PromptBuilder()

    def run(self, query: str) -> Dict[str, Any]:
        """
        Executes the full RAG flow: 
        Retrieve -> Build Context -> Prompt -> Generate -> Parse
        """
        logger.info(f"🚀 RAG Pipeline started for: {query}")

        # Retrieval (Get Top-N relevant chunks)
        # Note: We retrieve K candidates, re-rank them, and keep the best ones.
        hits = self.retrieval_pipeline.run(query=query, retrieve_k=100)

        if not hits:
            return {
                "answer": "هیچ سند مرتبطی در پایگاه داده یافت نشد.",
                "documents": {},
                "used_docs": []
            }

        # Context Building (Format them into [DOC_1] strings)
        context_str, doc_map = self.context_builder.build(hits)

        # Prompt Engineering
        messages = self.prompt_builder.build(query, context_str)

        # Generation
        answer = self.llm_client.generate(
            system_prompt=messages["system"],
            user_prompt=messages["user"]
        )

        # Citation Extraction (Simple heuristic)
        # We check which [DOC_X] tags appear in the final answer
        used_docs = []
        for doc_label in doc_map.keys():
            if doc_label in answer:
                used_docs.append(doc_label)

        return {
            "answer": answer,
            "documents": doc_map, # Map of DOC_1 -> Real Metadata
            "used_docs": used_docs
        }