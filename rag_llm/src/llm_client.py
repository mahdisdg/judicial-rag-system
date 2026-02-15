import os
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load env
load_dotenv()
logger = logging.getLogger("LegalRAG")

class LLMClient:
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        api_key: str = None,
        base_url: str = "https://api.avalai.ir/v1",
        **kwargs
    ):
        self.model_name = model_name
        
        # Store configuration
        self.config = {
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 1024),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }
        
        self.api_key = api_key or os.getenv("AVALAI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ API Key not found! LLM calls will fail.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Single turn generation"""
        return self._call_api([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    def generate_chat(self, messages: List[Dict[str, str]]) -> str:
        """Multi-turn generation"""
        return self._call_api(messages)

    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Pass **self.config to unpack all parameters
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.config 
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"❌ LLM API Error: {e}")
            return ""