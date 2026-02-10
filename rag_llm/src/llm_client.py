import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger("LegalRAG")

class LLMClient:
    """
    A wrapper for OpenAI-compatible APIs (like AvalAI).
    """
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        api_key: str = None,
        base_url: str = "https://api.avalai.ir/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Security check
        key = api_key or os.getenv("AVALAI_API_KEY")
        if not key:
            logger.warning("⚠️ API Key not found! LLM calls will fail.")
        
        self.client = OpenAI(
            api_key=key,
            base_url=base_url
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends the prompt to the LLM and returns the text response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"❌ LLM Generation Failed: {e}")
            return "Unfortunately, there was a problem with the language model."