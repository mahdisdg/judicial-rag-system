from typing import List, Dict
from .llm_client import LLMClient
from .logger import rag_logger

class QueryRewriter:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        
    def rewrite(self, query: str, history: List[Dict[str, str]]) -> str:
        if not history:
            rag_logger.info(f"🔄 [Rewriter] No history. Keeping original: '{query}'")
            return query

        rag_logger.info("🔄 [Rewriter] Processing query with history...")

        # Construct History String
        recent_history = history[-2:] 
        history_str = ""
        for msg in recent_history:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:200].replace("\n", " ") 
            history_str += f"{role}: {content}\n"

        rag_logger.debug(f"📜 [Rewriter] Context Used:\n{history_str.strip()}")

        system_prompt = """
You are an expert Persian Search Query Optimizer.
Rewrite the "Latest User Question" into a complete, standalone question.
1. Resolve Pronouns (he, she, it -> specific entity).
2. Restore Context from history.
3. Output ONLY the rewritten Persian question.

EXAMPLES:
History: User: مجازات سرقت؟ Assistant: حبس... User: مسلحانه چطور؟
>>> Rewritten: مجازات سرقت مسلحانه چیست؟
"""
        user_prompt = f"### History:\n{history_str}\n### Question:\n{query}\n### Rewritten:"
        
        rewritten = self.llm.generate(system_prompt, user_prompt)
        rewritten = rewritten.replace("Rewritten:", "").replace('"', '').strip()
        
        rag_logger.info(f"✅ [Rewriter] '{query}' -> '{rewritten}'")
        return rewritten