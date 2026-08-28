# Worker Agents Layer

Independent Python modules (not a fixed pipeline). Each agent follows the same template:

- `agent.py` — subclass of `BaseAgent`, implements `run(state)`
- `prompts.py` — LLM prompt templates (when needed)
- `tools.py` — shared-service integrations (Inference / RAG / Optimization)
- `config.py` — agent-local thresholds and paths

**Hard rule:** Agents never call Storage repositories. They only update `graph_state`.

## Current runtime profile

Agents continue to share the same `run(state) -> state` contract regardless of provider. The current configuration uses EVREN `llm-fast` for Classification and Extraction, EVREN `llm-large` for Summary and Writer, local Python rules for Validation/Routing, and local RAG. OCR remains independently configured and is intentionally not documented here as an EVREN dependency.

See `Documentation/architecture.md` §4 and `Documentation/agent_catalog.md`.
