from typing import Dict


class PromptBuilder:
    """
    Strict RAG prompt builder.
    - Instructions in English
    - Output language strictly enforced as Persian (Farsi)
    - Citations are mandatory
    - Hallucination is forbidden
    """

    SYSTEM_PROMPT = (
        "You are a professional legal assistant.\n"
        "You MUST follow all rules strictly.\n"
        "You MUST answer ONLY using the provided sources.\n"
        "You MUST NOT use any external or prior knowledge.\n\n"
        "LANGUAGE RULE:\n"
        "- The final answer MUST be written ONLY in Persian (Farsi).\n"
        "- Do NOT use English or any other language in the answer.\n\n"
        "CITATION RULES:\n"
        "- Every factual statement MUST have a citation.\n"
        "- Citations must use this exact format: [DOC_X].\n"
        "- Answers without citations are INVALID.\n\n"
        "INSUFFICIENT INFORMATION RULE:\n"
        "- If the sources do not explicitly contain the answer, respond ONLY with:\n"
        "\"اطلاعات کافی در منابع ارائه‌شده وجود ندارد.\""
    )

    def build(self, query: str, context: str) -> Dict[str, str]:
        """
        Build prompt messages for LLM.

        Args:
            query: User question
            context: Context built by ContextBuilder

        Returns:
            Messages dict for chat completion
        """

        user_prompt = (
            "### QUESTION:\n"
            f"{query}\n\n"
            "### SOURCES:\n"
            f"{context}\n\n"
            "### ANSWERING INSTRUCTIONS:\n"
            "- Answer ONLY based on the sources\n"
            "- Write the answer ONLY in Persian (Farsi)\n"
            "- Use formal legal Persian\n"
            "- Add at least one citation [DOC_X] per paragraph\n"
            "- Do NOT guess or hallucinate\n\n"
            "### FINAL ANSWER (Persian only):"
        )

        return {
            "system": self.SYSTEM_PROMPT,
            "user": user_prompt,
        }
