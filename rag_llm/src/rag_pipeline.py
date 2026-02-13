from typing import Dict, Any, List
from retrieval.src.pipeline import RetrievalPipeline
from retrieval.src.context_builder import ContextBuilder
from .llm_client import LLMClient
from .prompt import PromptBuilder
from .query_rewriter import QueryRewriter
from .logger import rag_logger

class RAGPipeline:
    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        llm_client: LLMClient,
        max_docs_in_context: int = 8
    ):
        self.retrieval_pipeline = retrieval_pipeline
        self.llm_client = llm_client
        self.context_builder = ContextBuilder(max_docs=max_docs_in_context)
        self.prompt_builder = PromptBuilder()
        self.rewriter = QueryRewriter(llm_client)

    def run(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        if chat_history is None: chat_history = []

        rag_logger.info("="*50)
        rag_logger.info(f"▶️ START PIPELINE: {query}")

        # Rewrite (USES HISTORY)
        search_query = self.rewriter.rewrite(query, chat_history)

        # Retrieve
        hits = self.retrieval_pipeline.run(query=search_query, retrieve_k=50)

        # Build Context
        context_str, doc_map = self.context_builder.build(hits)
        
        if context_str:
            rag_logger.debug(f"📦 [Context] Length: {len(context_str)} chars")
            rag_logger.debug(f"📦 [Context Preview]: {context_str[:500]}...")
        else:
            rag_logger.warning("⚠️ [Context] Context is EMPTY!")

        # Prompt Engineering
        system_msg = self.prompt_builder.SYSTEM_PROMPT
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Use the REWRITTEN query + New Context
        augmented_user_msg = self.prompt_builder.build_user_message(search_query, context_str)
        messages.append({"role": "user", "content": augmented_user_msg})

        # Generate
        rag_logger.info("🤖 [LLM] Generating answer...")
        answer = self.llm_client.generate_chat(messages)
        rag_logger.debug(f"🤖 [LLM Output] {answer[:200]}...")

        # Citations
        used_docs = [doc_id for doc_id in doc_map.keys() if doc_id in answer]
        
        rag_logger.info("🏁 END PIPELINE")
        return {
            "original_query": query,
            "rewritten_query": search_query,
            "answer": answer,
            "documents": doc_map,
            "used_docs": used_docs
        }