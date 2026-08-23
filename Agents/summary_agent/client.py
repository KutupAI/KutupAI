"""Thin Inference wrapper used by SummaryAgent."""

from __future__ import annotations

from dataclasses import dataclass

from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.inference_response import InferenceResponse
from Inference.client.llama_client import LlamaClient

from .config import SummaryConfig


@dataclass(frozen=True)
class SummaryRequest:
    prompt: str


class SummaryClient:
    """Routes prompts to Gemma 3 via LlamaClient / llama-server."""

    def __init__(
        self,
        config: SummaryConfig | None = None,
        llama_client: LlamaClient | None = None,
    ) -> None:
        self.config = config or SummaryConfig.from_env()
        self._llama = llama_client or LlamaClient(
            base_url=self.config.inference_url,
            timeout=self.config.timeout,
        )

    def generate(self, request: SummaryRequest) -> InferenceResponse:
        return self._llama.generate(
            InferenceRequest(
                messages=[Message(role="user", content=request.prompt)],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )
