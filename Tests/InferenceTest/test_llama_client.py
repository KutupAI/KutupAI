import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Inference.client.inference_request import InferenceRequest, Message
from Inference.client.llama_client import LlamaClient

client = LlamaClient()

request = InferenceRequest(
    messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello"),
    ]
)

response = client.generate(request)

print(response.success)
print(response.text)
print(response.error)