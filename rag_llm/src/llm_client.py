from typing import Optional
import logging
import os

from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once
load_dotenv()


class AvalAIClient:
    """
    LLM client implementation using AvalAI (OpenAI-compatible API).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        base_url: str = "https://api.avalai.ir/v1",
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.getenv("AVALAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "AVALAI_API_KEY not found. Please set it in .env or environment variables."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info("[LLM] AvalAIClient initialized")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response from AvalAI.
        """

        logger.info("[LLM] Sending request to AvalAI")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        request_id = getattr(response, "_request_id", None)
        if request_id:
            logger.info(f"[LLM] AvalAI request_id: {request_id}")

        output_text = response.choices[0].message.content.strip()

        logger.info("[LLM] Response received from AvalAI")

        return output_text
