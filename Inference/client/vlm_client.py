"""
vlm_client.py
--------------------
Client for the local Vision-Language Model server, used only by
classification_agent (see Documentation/architecture.md addition:
"multi-model Inference" -- Gemma3 stays the default text LLM for every
other Agent; a VLM is added specifically because classification needs
image + text + layout together, per the task document section 4).

Renamed from qwen_vl_client.py (2024): the model behind this endpoint
switched from Qwen2.5-VL to Gemma 3 (4B/12B/27B, served locally via
llama.cpp/llama-server) -- Qwen VL was several generations old and Gemma 3
now natively supports vision, matching the Gemma3-everywhere-else policy
already in place for every other Agent (see module docstring above). None
of the code below is model-specific (plain OpenAI-compatible chat-
completions protocol), so this rename is a naming/clarity change only --
no wire-format change.

Mirrors llama_client.py's shape/contract on purpose (same request/response
style, same OpenAI-compatible chat-completions protocol) so Agents-layer
code stays consistent across both clients. The only real difference is
that messages can carry an image (base64 data URL) alongside text, using
the same multimodal content-array format llama.cpp's server and most
OpenAI-compatible VLM servers accept.

Backward compatibility: `QwenVLClient` / `QwenVLRequest` are kept as
aliases at the bottom of this file so existing imports (evaluation/
ablation.py etc.) keep working without an immediate edit. New code should
import `VLMClient` / `VLMRequest` from here instead.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import requests

from .inference_response import InferenceResponse


@dataclass(slots=True)
class VLMRequest:
    text_prompt: str
    system_prompt: str | None = None
    image_bytes: bytes | None = None
    image_media_type: str = "image/png"
    # Optional model identifier forwarded in the request payload. A
    # single-model llama-server instance ignores this field, but sending
    # it is harmless and future-proofs against a multi-model llama-server
    # / vLLM setup that routes by model name.
    model: str | None = None
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False


class VLMClient:
    """Thin HTTP client for a vision-capable OpenAI-compatible server
    (llama.cpp multimodal server serving Gemma 3 + its mmproj vision
    projector, vLLM, or similar). Kept separate from LlamaClient/the
    plain-text Gemma3 client so replacing/upgrading the classification
    model does not touch the client every other Agent depends on.

    IMPORTANT (llama.cpp deployment note): Gemma 3's vision support
    requires starting llama-server with BOTH the Gemma 3 GGUF (--model)
    AND its matching multimodal projector GGUF (--mmproj, e.g.
    mmproj-model-f16.gguf from the same release) -- without --mmproj the
    server will silently ignore image_url content and answer text-only.
    """

    def __init__(
        self,
        # Same shared Inference endpoint as LlamaClient / summary_agent
        # (Inference/llama_server → gemma3.gguf on :8080).
        base_url: str = "http://127.0.0.1:8111/v1/chat/completions",
        timeout: int = 300,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, request: VLMRequest) -> InferenceResponse:
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
            # Same as LlamaClient: keep answers in message.content for Gemma-4.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if request.model:
            payload["model"] = request.model

        try:
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            message = data["choices"][0]["message"]
            content = message.get("content")
            if not (isinstance(content, str) and content.strip()):
                reasoning = message.get("reasoning_content")
                content = reasoning if isinstance(reasoning, str) else (content or "")

            return InferenceResponse(
                success=True,
                text=content if isinstance(content, str) else str(content or ""),
                model=data.get("model"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason"),
            )
        except Exception as ex:
            return InferenceResponse(success=False, text="", error=str(ex))


# --- Backward-compat aliases -------------------------------------------------
# Existing imports like
#   from Inference.client.qwen_vl_client import QwenVLClient, QwenVLRequest
# will break once qwen_vl_client.py is deleted (see CHANGES notes on which
# callers to update: classification_agent/tools.py). These aliases exist so
# any OTHER, not-yet-updated caller keeps working during the migration.
QwenVLClient = VLMClient
QwenVLRequest = VLMRequest