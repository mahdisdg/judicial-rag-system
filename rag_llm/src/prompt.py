class PromptBuilder:
    
    SYSTEM_PROMPT = """
You are a professional "Smart Judicial Assistant" (دستیار هوشمند قضایی).
Answer the user's legal question based ONLY on the provided context chunks.

### RULES:
1. **Source of Truth:** Answer ONLY based on the [Context] provided in the last message.
2. **Language:** Formal Persian (Farsi).
3. **Citations:** Cite source ID [DOC_X] for every claim.
4. **Honesty:** If context is insufficient, say "اطلاعات کافی در اسناد یافت نشد".
"""

    def build_user_message(self, query: str, context_str: str) -> str:
        return f"""
### Context (Legal Documents):
{context_str}

### Question:
{query}

### Instructions:
Answer based on the context above. Cite [DOC_X].
"""