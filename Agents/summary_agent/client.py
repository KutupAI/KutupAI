"""Thin Inference wrapper used by SummaryAgent."""

from __future__ import annotations

from dataclasses import dataclass

from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.inference_response import InferenceResponse
from Inference.client.evren_client import EvrenClient
from Inference.client.llama_client import LlamaClient

from .config import SummaryConfig


@dataclass(frozen=True)
class SummaryRequest:
    prompt: str


class SummaryClient:
    """Özet isteklerini seçilen yerel veya EVREN çıkarım istemcisine yollar."""

    def __init__(
        self,
        config: SummaryConfig | None = None,
        llama_client: LlamaClient | None = None,
    ) -> None:
        self.config = config or SummaryConfig.from_env()
        if llama_client is not None:
            self._client = llama_client
        elif self.config.inference_backend == "evren":
            self._client = EvrenClient(model=self.config.evren_model, timeout=self.config.timeout)
        else:
            self._client = LlamaClient(
                base_url=self.config.inference_url,
                timeout=self.config.timeout,
            )

    def generate(self, request: SummaryRequest) -> InferenceResponse:
        return self._client.generate(
            InferenceRequest(
                messages=[Message(role="user", content=request.prompt)],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )
