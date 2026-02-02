from __future__ import annotations

import os
from typing import Optional, TypeVar

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"


ModelT = TypeVar("ModelT")


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=key)

    def generate_json(self, prompt: str, schema_model: type[ModelT]) -> ModelT:
        schema_dict = schema_model.model_json_schema()
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema_dict,
                temperature=0.2,
            ),
        )
        raw_text = (response.text or "").strip()
        try:
            return schema_model.model_validate_json(raw_text)
        except Exception as exc:  # pragma: no cover - depends on Gemini response
            raise ValueError(f"Failed to parse structured output: {raw_text}") from exc

    def generate_json_with_image(
        self, prompt: str, image_bytes: bytes, schema_model: type[ModelT]
    ) -> ModelT:
        schema_dict = schema_model.model_json_schema()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema_dict,
                temperature=0.2,
            ),
        )
        raw_text = (response.text or "").strip()
        try:
            return schema_model.model_validate_json(raw_text)
        except Exception as exc:  # pragma: no cover - depends on Gemini response
            raise ValueError(f"Failed to parse structured output: {raw_text}") from exc

    def generate_text(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
            ),
        )
        return response.text or ""
