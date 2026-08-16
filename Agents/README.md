# Worker Agents Layer

Independent Python modules (not a fixed pipeline). Each agent follows the same template:

- `agent.py` — subclass of `BaseAgent`, implements `run(state)`
- `prompts.py` — LLM prompt templates (when needed)
- `tools.py` — shared-service integrations (Inference / RAG / Optimization)
- `config.py` — agent-local thresholds and paths

**Hard rule:** Agents never call Storage repositories. They only update `graph_state`.

See `Documentation/architecture.md` §4 and `Documentation/agent_catalog.md`.
