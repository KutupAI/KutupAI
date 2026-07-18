from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient

client = LlamaClient()

request = InferenceRequest(
    messages=[
        Message(
            role="system",
            content="You are a helpful assistant."
        ),
        Message(
            role="user",
            content="Hello"
        )
    ]
)

response = client.generate(request)#هون التنفيذ كامل الارسال والاستقبال 

print(response.success)
print(response.text)
print(response.error)