# summary_agent

## Purpose

Pipeline stage between RAG and writer: grounded summary notes for the final draft.

```text
… → RAG → SummaryAgent → Routing → WriterAgent → …
```

No model loading — Inference only via `SummaryClient` → `LlamaClient`.

## Orchestration contract

`SummaryAgent.run(state)` is the graph entry point.

| Direction | Key | Shape |
|---|---|---|
| Read | `request.question` (or top-level `question`) | `str` |
| Read | `rag` **or** `rag_result` | RAG payload (see below) |
| Write | `summary` | `{success, rag_summary_text, error?}` |

**RAG input** (any of these):

- Pipeline: `state["rag"] = {success, rag_data: {query, results: [...]}, error?}`
- GraphState: `state["rag_result"] = {success, data: {query, results: [...]}, error?}`
- Simplified rag_agent: `state["rag_result"] = {context, sources, result_count}`

**Summary output:**

```json
{ "success": true, "rag_summary_text": "- … [6458/31]" }
```

```json
{ "success": false, "rag_summary_text": null, "error": {"code": "…", "message": "…"} }
```

Downstream: `writer_agent` / `routing_agent` read `summary.rag_summary_text`.

## Standalone API

`summarize(question, rag_result)` and `tools.summarize_context(...)` use the RAGResult envelope (`success` / `data` / `error`) without GraphState.

## Files

| File | Role |
|---|---|
| `agent.py` | `SummaryAgent` — validate → prompt → Inference → `state["summary"]` |
| `client.py` | `SummaryClient` over `LlamaClient` |
| `prompts.py` | Deterministic prompt |
| `schemas.py` | RAG in / summary out models |
| `config.py` | Env + model registry |
| `tools.py` | Standalone helper |
| `mock_data.py` | Fixtures for manual/tests |

## Config

`Orchestration/config.yaml`:

```yaml
summary:
  enabled: true   # flip when RAG is enabled in the same config
  module: Agents.summary_agent
  class_name: SummaryAgent
  fallback: skip
```

## Summary Agent
```bash
# from repo root — live llama-server (Gemma)
python Tests/Agents/manual_test.py

# mocked Integration with Orchestration
pytest Orchestration/tests/test_summary_integration.py -q
```
