"""
Minimal Gemini client for CLUES framework.

Provides async LLM access for similarity scoring and disambiguation.
Uses the google-genai SDK with structured output support via Pydantic schemas.
"""

import json
import logging
import os

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from typing import Optional, Type, Union

logger = logging.getLogger(__name__)

# Load .env from the clues project root
load_dotenv()


class GeminiClient:
    """Minimal async Gemini client for CLUES."""

    def __init__(self, model_name: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or as an environment variable."
            )
        self.client = genai.Client(api_key=self.api_key)

    async def get_response(
        self,
        prompt: str,
        response_schema: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Get a response from Gemini.

        Args:
            prompt: The prompt to send.
            response_schema: Optional Pydantic model for structured output.

        Returns:
            str or Pydantic model instance.
        """
        config = {}

        if response_schema:
            schema_dict = response_schema.model_json_schema()
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema_dict

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config if config else None,
        )

        response_text = response.text

        if response_schema:
            parsed_data = json.loads(response_text)
            return response_schema(**parsed_data)

        return response_text
