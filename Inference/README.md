# AI Inference Layer

Replaceability: to swap Gemma 3 for another model later, it is enough to change models/ + client/llama_client.py internals; the external interface (the functions Agents call) stays the same.

## Current provider routing

`Inference/client/evren_client.py` is the OpenAI-compatible client used by the active EVREN profile. The agent-level environment variables select the provider and model without changing the call contracts:

| Agent | Default backend | Default model |
|---|---|---|
| Classification | EVREN | `llm-fast` |
| Extraction | EVREN | `llm-fast` |
| Summary | EVREN | `llm-large` |
| Writer | EVREN | `llm-large` |

The local llama.cpp client remains available as a development fallback. API keys belong only in the project-root `.env`; never place them in this README or commit them to Git.
