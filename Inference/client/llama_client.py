import requests#مكتبة لعمل طلبات HTTP
import requests

from .inference_request import InferenceRequest
from .inference_response import InferenceResponse

class LlamaClient:

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1/chat/completions",
        timeout: int = 300,#إذا لم يجب Gemma خلال 300 ثانية فشل الاتصال يعطيه 
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

            return InferenceResponse(
                success=False,
                text="",
                error=str(ex),
            )