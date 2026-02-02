from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"


@dataclass
class GeminiResult:
    data: Dict[str, Any]
    raw_text: str


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=key)

    def generate_json(self, prompt: str, response_schema: Dict[str, Any]) -> GeminiResult:
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2,
            ),
        )
        raw_text = response.text or ""
        return GeminiResult(data=self._safe_json(raw_text), raw_text=raw_text)

    def generate_json_with_image(
        self, prompt: str, image_bytes: bytes, response_schema: Dict[str, Any]
    ) -> GeminiResult:
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.2,
            ),
        )
        raw_text = response.text or ""
        return GeminiResult(data=self._safe_json(raw_text), raw_text=raw_text)

    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )
        return response.text or ""

    @staticmethod
    def _safe_json(raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {}
