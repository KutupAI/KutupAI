# summary_agent

## Purpose

`summary_agent` sits between `rag_agent` and `writer_agent` in the Orchestration
workflow. It takes the user's original question and the RAG Layer's retrieval
result, sends both to Gemma 3 (via `Inference/client/llama_client.py`), and
returns a concise, source-grounded summary that `writer_agent` uses to draft
the official response.

It contains no model logic. It never calls llama.cpp, loads GGUF files, or
manages GPU/KV-cache resources — all of that stays inside the Inference Layer
(`Inference/models/gemma3.gguf` + `Inference/llamastart.bat`).
`summary_agent` only builds a prompt and calls `SummaryClient.generate(...)`.

## Input / Output

**Input**

- `question: str` — the user's original question.
- `rag_result: dict` — the RAG Layer's output, matching its existing contract:

```json
{
  "success": true,
  "data": {
    "operation": "retrieve",
    "query": "...",
    "document_id": "doc-6458-001",
    "file_name": "...",
    "results": [
      {
        "chunk_id": "...",
        "text": "...",
        "law_number": "6458",
        "article_no": "31",
        "article_type": "madde",
        "page_start": 12,
        "page_end": 13,
        "score": 0.91
      }
    ]
  },
  "error": null
}
```

**Output** — same envelope shape as the RAG contract, so `writer_agent` and
Orchestration handle both uniformly:

```json
{
  "success": true,
  "data": {
    "operation": "summarize_context",
    "query": "...",
    "summary": "...",
    "sources": [
      {
        "chunk_id": "...",
        "law_number": "6458",
        "article_no": "31",
        "article_type": "madde",
        "page_start": 12,
        "page_end": 13
      }
    ]
  },
  "error": null
}
```

On any failure, `success` is `false` and `error` is `{"code": ..., "message": ...}`.
Handled failure codes: `invalid_input`, `rag_failed`, `empty_context`,
`inference_error`, `invalid_model_response`. The model is never called when
`rag_result.success` is `false` or `results` is empty.

## Processing Flow

```
question + rag_result
      │
      ▼
validate input (schemas.py)
      │
      ▼
build_prompt(question, results)   (prompts.py)
      │
      ▼
SummaryClient.generate()          (client.py)
      │
      ▼
LlamaClient → llama-server        (Inference/client/llama_client.py)
      │
      ▼
Gemma 3 (Inference/models/gemma3.gguf)
      │
      ▼
validate model output → SummaryAgentResult
```

`SummaryAgent.run(state)` is the Orchestration entry point: it reads
`state["question"]` / `state["rag_result"]` and writes `state["summary_result"]`.
`SummaryAgent.summarize(question, rag_result)` is the same logic usable
directly (e.g. for manual verification), independent of graph state.

## Files

| File | Responsibility |
|---|---|
| `agent.py` | `SummaryAgent(BaseAgent)` — validation → prompt → inference → output. |
| `client.py` | `SummaryClient` — thin wrapper over `LlamaClient`. |
| `prompts.py` | Deterministic prompt template and `build_prompt()`. |
| `schemas.py` | Pydantic models for RAG input and summary output contracts. |
| `config.py` | Agent inference params; reads `Inference/models/model_registry.json`. |
| `tools.py` | `summarize_context()` helper for Orchestration. |
| `mock_data.py` | Sample RAG result for manual checks (not used in production). |
| `manual_test.py` | Manual runner against a live llama-server. |

## Manual verification

```python
from unittest.mock import MagicMock

from Inference.client.inference_response import InferenceResponse
from Agents.summary_agent.agent import SummaryAgent
from Agents.summary_agent.client import SummaryClient
from Agents.summary_agent.mock_data import MOCK_QUESTION, MOCK_RAG_RESULT

mock_llama = MagicMock()
mock_llama.generate.return_value = InferenceResponse(
    success=True,
    text="- İkamet izni başvurusu valiliğe yapılır. [6458/31]",
)

agent = SummaryAgent(client=SummaryClient(llama_client=mock_llama))
result = agent.summarize(MOCK_QUESTION, MOCK_RAG_RESULT)
print(result.model_dump())
```

## Live test (Gemma 3)

1. Place `gemma3.gguf` in `Inference/models/` (see `Inference/models/README.md`).
2. Start the server: `Inference/llamastart.bat`
3. Run: `python -m Agents.summary_agent.manual_test`

## Automated tests

```bash
pip install -r Agents/summary_agent/requirements.txt
python Tests/Agents/test_summary_agent.py
# or
pytest Tests/Agents/test_summary_agent.py -q
```

Tests mock `LlamaClient` — they do **not** require a running llama-server or GGUF weights.
2. اختبار يدوي — يطبع JSON في التيرمنال
python -m Agents.summary_agent.manual_test
3. اختبار آلي — من Tests

python Tests/Agents/test_summary_agent.py