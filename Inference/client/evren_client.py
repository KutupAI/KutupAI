"""EVREN'in OpenAI uyumlu çıkarım API'si için hafif istemci."""

from __future__ import annotations

import os
from typing import Optional

import requests
from dotenv import load_dotenv

from .inference_request import InferenceRequest
from .inference_response import InferenceResponse
from .llama_client import _message_text


class EvrenClient:
    """Yerel llama-server yerine EVREN'de seçili bir modeli çağırır."""

    def __init__(self, *, model: str, timeout: int = 120) -> None:
        load_dotenv()
        base_url = os.getenv("EVREN_LLM_BASE_URL", "").strip().rstrip("/")
        api_key = os.getenv("EVREN_LLM_API_KEY", "").strip()
        if not base_url or not api_key:
            raise RuntimeError("EVREN_LLM_BASE_URL ve EVREN_LLM_API_KEY .env içinde tanımlı olmalıdır.")
        self.base_url = f"{base_url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            message = data["choices"][0]["message"]
            return InferenceResponse(
                success=True,
                text=_message_text(message),
                model=data.get("model") or self.model,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except Exception as exc:  # API hatası Writer sözleşmesinde success=false olur.
            return InferenceResponse(success=False, text="", error=str(exc))
