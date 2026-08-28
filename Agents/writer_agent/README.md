# Writer Agent

## Güncel proje ayarı

Writer, nihai Türkçe yanıt için varsayılan olarak EVREN `llm-large` kullanır: `WRITER_INFERENCE_BACKEND=evren`, `WRITER_EVREN_MODEL=llm-large`. Bu seçim yalnız `Inference` istemcisini değiştirir; Writer'ın `writing.answer` çıktısı ve katmanlar arası sözleşme değişmez.

Bu dosyadaki yerel `llama-server` akışı alternatif geliştirme modudur; EVREN yapılandırması etkin olduğunda günlük çalıştırmada gerekli değildir.

## Purpose

Generates the final natural-language answer to the user's question,
using only the relevant parts of the Unified State, and writes it to
`state["writing"]`. It is the last content-producing step before
Presentation.

```text
Presentation -> Application -> Orchestration -> Writer Agent -> Inference -> EVREN llm-large (or local fallback)
```

## Responsibility and boundaries

Writer Agent:

1. Receives the full Unified State.
2. Extracts only the fields relevant to answering the question.
3. Builds a compact prompt (system instruction + short dynamic context).
4. Calls the existing Inference client.
5. Writes the result to `writing.answer`.
6. Leaves every other section of the Unified State untouched.

Writer Agent never:

- decides what Presentation displays;
- controls Orchestration;
- starts, launches, or configures a llama-server;
- loads `gemma3.gguf` or any model file directly;
- implements its own inference/runtime code.

All model access goes exclusively through the project's existing Inference
layer (`Inference/client/evren_client.py` or `Inference/client/llama_client.py`).
This layer is not modified or duplicated here.

## File structure

```text
Agents/writer_agent/
├── agent.py          # WriterAgent class: extract -> prompt -> Inference -> writing
├── prompts.py         # Fixed system instruction + dynamic prompt builder
├── schemas.py          # Internal dataclasses: WriterContext, WritingResult
├── config.py           # Generation params (temperature, max_tokens, ...)
├── tools.py             # Optional legal-grounding helper (not used by default)
├── README.md
└── requirements.txt
```

## Input: Unified State contract

Writer Agent reads (all optional/defensive-checked, missing sections
are treated as empty):

| Field | Used for |
|---|---|
| `request.question` | the question to answer |
| `classification.document_type` | context label |
| `summary.rag_summary_text` | grounded context (already retrieved upstream) |
| `extraction.*` | relevant extracted fields, only if non-empty |
| `validation.is_complete` / `.errors` / `.warnings` | only surfaced if actionable |

It does **not** forward raw `ocr.ocr_data` or raw `rag.rag_data` to the
model when `summary.rag_summary_text` already covers it, to keep the
prompt small.

## Output contract

Writer Agent updates only the `writing` key:

```json
"writing": { "success": true, "answer": "..." }
```

On any failure (missing question, Inference failure, unexpected
exception):

```json
"writing": { "success": false, "answer": "" }
```

No other section of the Unified State is added, removed, or modified.

## Connection to Inference

```text
WriterAgent._call_inference()
    -> selected provider client
    -> EVREN `llm-large` (default) or local llama-server (fallback)
```

The selected client is constructed with its configuration defaults unless a
client is injected (used in tests). Writer Agent never touches
`llama_server/`, `model_registry.json`, or any model file.

## Usage from Orchestration

```python
from Agents.writer_agent.agent import WriterAgent

writer = WriterAgent()
state = writer.run(state)  # state["writing"] is now populated
```

The same `WriterAgent` class is used by Orchestration and by the
standalone test below — no separate implementation exists for either.

> **Note:** `Agents/base/base_agent.py` was not available at
> implementation time. `agent.py` imports `BaseAgent` from
> `Agents.base.base_agent` and falls back to a minimal local
> placeholder only if that import fails, so the file can still be
> exercised in isolation. Once integrated, confirm the real
> `BaseAgent` matches the assumed `run(state) -> state` contract; no
> other change should be needed.

## Running the standalone test

```bash
python -m unittest Tests.Agents.test_writer_agent -v
```

(run from the project root, so `Agents` and `Inference` are importable)

## Live standalone test

With the active backend configured (EVREN by default, or a local llama-server), run:

```bash
python Tests/Agents/manual_writer_live.py
```

To use your own Unified State JSON:

```bash
python Tests/Agents/manual_writer_live.py --input path/to/writer_state.json
```

## What the test verifies

- a valid Unified State is accepted and produces `writing.success = True`
  and a non-empty `writing.answer`;
- the question and `summary.rag_summary_text` are correctly included in
  the prompt sent to Inference;
- the selected Inference client is called, not a new/duplicate inference path;
- an Inference failure results in `writing = {"success": False, "answer": ""}`;
- a missing question short-circuits without calling Inference at all;
- a non-dict state raises `TypeError` immediately.

Only the network call inside the selected provider client is mocked — no
live provider needs to run to execute this test. Running it against the
configured backend (no mocking) exercises the exact same code path
Orchestration uses.

## Configuration / dependencies

No configuration beyond `config.py`'s generation parameters
(`TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `MAX_SUMMARY_CHARS`). Model host,
port, and model file are entirely owned by the Inference layer's own
configuration and are not duplicated here. See `requirements.txt` for
external dependencies (none beyond the standard library).

layer test. Running it
python Tests/Agents/manual_writer_live.py
