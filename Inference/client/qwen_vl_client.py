"""
qwen_vl_client.py
--------------------
Client for the Qwen Vision-Language Model server, used only by
classification_agent (see Documentation/architecture.md addition:
"multi-model Inference" -- Gemma3 stays the default text LLM for every
other Agent; Qwen VLM is added specifically because classification needs
image + text + layout together, per the task document section 4).

Mirrors llama_client.py's shape/contract on purpose (same request/response
style, same OpenAI-compatible chat-completions protocol) so Agents-layer
code stays consistent across both clients. The only real difference is
that messages can carry an image (base64 data URL) alongside text, using
the same multimodal content-array format llama.cpp's server and most
OpenAI-compatible VLM servers accept.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import requests

from .inference_response import InferenceResponse


@dataclass(slots=True)
class QwenVLRequest:
    text_prompt: str
    system_prompt: str | None = None
    image_bytes: bytes | None = None
    image_media_type: str = "image/png"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False


class QwenVLClient:
    """Thin HTTP client for a Qwen-VL-capable OpenAI-compatible server
    (llama.cpp multimodal server, vLLM, or similar). Kept separate from
    LlamaClient/gemma3 so replacing/upgrading the classification model does
    not touch the client every other Agent depends on.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8092/v1/chat/completions",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, request: QwenVLRequest) -> InferenceResponse:
        messages: list[dict] = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        if request.image_bytes:
            encoded = base64.b64encode(request.image_bytes).decode("ascii")
            data_url = f"data:{request.image_media_type};base64,{encoded}"
            user_content = [
                {"type": "text", "text": request.text_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        else:
            # Text-only fallback (e.g. OCR confidence so low there is no
            # usable rendered page, or image unavailable) -- ablation tests
            # in the task document section 10 explicitly need this path
            # ("OCR text only") to exist and be comparable.
            user_content = request.text_prompt

        messages.append({"role": "user", "content": user_content})

        payload = {
            "messages": messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})

            return InferenceResponse(
                success=True,
                text=data["choices"][0]["message"]["content"],
                model=data.get("model"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except Exception as ex:
            return InferenceResponse(success=False, text="", error=str(ex))
