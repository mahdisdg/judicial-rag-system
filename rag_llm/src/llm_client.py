import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from .logger import rag_logger

load_dotenv()

class LLMClient:
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, **kwargs):
        self.model_name = model_name
        key = api_key or os.getenv("AVALAI_API_KEY")
        if not key: rag_logger.error("API Key missing!")
        self.client = OpenAI(api_key=key, base_url="https://api.avalai.ir/v1")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._call_api([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        return self._call_api(messages)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            rag_logger.error(f"❌ API Error: {e}")
            return "خطای ارتباط با سرور."