import requests  # مكتبة لعمل طلبات HTTP

from .inference_request import InferenceRequest
from .inference_response import InferenceResponse


def _message_text(message: dict) -> str:
    """Gemma-4 may put the answer in reasoning_content when thinking eats max_tokens."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if content is None:
        return ""
    return str(content)


class LlamaClient:

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082/v1/chat/completions",
        timeout: int = 300,  # إذا لم يجب Gemma خلال 300 ثانية فشل الاتصال يعطيه
    ):
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:

        payload = {
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            # Disable Gemma-4 "thinking" so the answer lands in message.content
            # (otherwise reasoning burns the token budget and content stays empty).
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:

            response = requests.post(
                self.base_url,
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
                model=data.get("model"),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=data["choices"][0].get("finish_reason"),
            )

        except Exception as ex:

            return InferenceResponse(
                success=False,
                text="",
                error=str(ex),
            )
