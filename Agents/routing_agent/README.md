# Routing Agent

A production-grade, explainable document routing agent for the KutupAI
pipeline. Given a document (plus everything the upstream OCR /
classification / extraction / validation / RAG / summary agents already
produced), it determines which department should handle it.

Routing is never a single LLM call. It's a weighted combination of eight
independent signals:

| Signal    | What it checks                                            |
|-----------|------------------------------------------------------------|
| rule      | department `routing_rules` / `excluded_topics` matches     |
| keyword   | keyword / handled-topic overlap                            |
| bm25      | classic BM25 relevance over department corpora             |
| semantic  | token-overlap cosine similarity (pluggable → real embeddings) |
| metadata  | institution / recipient / correspondence-history alignment |
| legal     | legal reference vs. department legal authority overlap     |
| entity    | named entity overlap                                       |
| llm       | optional LLM-in-the-loop judgement (off by default)         |

...followed by deterministic hierarchy resolution, authority resolution,
conflict detection, ambiguity detection, and confidence calibration.

## Requirements

None. The core agent runs on the Python standard library only
(`dataclasses`, `enum`, `abc`, `typing`, `re`, `math`, `collections`).
Python 3.9+ recommended. See `requirements.txt` for optional integrations
(real LLM scoring, real embeddings) that are **not** required to run or
import the package.

## Installation

Drop the `Agents/` directory into your project root (next to wherever your
orchestrator lives) so it's importable as `Agents.routing_agent`:

```
your_project/
├── Agents/
│   ├── __init__.py
│   └── routing_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── config.py
│       ├── knowledge_base.py
│       ├── models.py
│       ├── prompts.py
│       ├── tools.py
│       ├── validators.py
│       └── tests/
├── requirements.txt
└── README.md
```

No `pip install` needed for the base functionality.

## Pipeline contract

The agent consumes and produces the shared pipeline envelope:

```
{request, ocr, classification, extraction, validation, rag, summary, routing, writing}
```

**Input** — `routing` arrives empty (`{}`); every other stage may be
partially or fully populated, and any stage may be `{}` or have
`"success": false`. The agent never crashes on partial upstream data and
never requires every upstream stage to have succeeded.

**Output** — the exact same envelope is returned, with only `routing`
replaced:

```json
"routing": {
  "success": true,
  "department": "Bilgi İşlem Daire Başkanlığı"
}
```

`success` is `false` only when no department could be determined at all
(fully empty document text/question, an internally inconsistent decision
caught by post-hoc validation, or zero candidates in the knowledge base).
Low confidence, ambiguity, or upstream signal conflicts still produce a
best-effort department with `success: true` — a low-confidence route is
still a routed decision, not a failed one.

## Usage

### Quick start — functional API

```python
from Agents.routing_agent import process

envelope = {
    "request": {
        "success": True,
        "question": "bu ne sozlesmesi",
        "document": {"document_id": "DOC-001", "file_name": "Elektrik sozlesmesi.pdf", "file_type": "pdf"},
    },
    "ocr": {"success": True, "ocr_data": {"page_count": 1, "language": "tr", "pages": [],
             "full_text": "...", "vision": {"signature": {"detected": True, "handwritten": True},
             "stamp": {"detected": False}}}},
    "classification": {"success": True, "document_type": "Elektrik sozlesmesi", "classification_confidence": 0.95},
    "extraction": {"success": True, "sender": None, "date": None, "address": None, "phone": None, "email": None},
    "validation": {"success": True, "is_complete": False, "errors": [], "warnings": []},
    "rag": {"success": True, "rag_data": {"operation": "retrieve", "query": "...", "results": []}},
    "summary": {"success": True, "rag_summary_text": "..."},
    "routing": {},
    "writing": {},
}

envelope = process(envelope)
print(envelope["routing"])  # {"success": True, "department": "..."}
```

### Using your own `RoutingAgent` instance

Use this when you need a custom knowledge base, a real semantic scorer, or
a real LLM scorer, or want to reuse one agent instance across many calls
explicitly instead of relying on the shared default.

```python
from Agents.routing_agent import RoutingAgent, default_knowledge_base

agent = RoutingAgent(knowledge_base=default_knowledge_base())
envelope = agent.process(envelope)
```

### Orchestration / GraphState

The workflow adapter calls `agent.run(state)`. That method adapts GraphState
(`ocr_result`, `classification_result`, `rag_result`, `summary`, …) into the
pipeline envelope, then writes only:

```python
state["routing"] = {"success": True, "department": "..."}
```

```bash
# real agent + mocked upstream stages
pytest Orchestration/tests/test_routing_integration.py -q
```

Enable in `Orchestration/config.yaml` (`agents.routing.enabled: true`) once
upstream stages provide usable document text / summary. Until then, pass
`RoutingAgent()` via `agent_overrides` in tests (see
`Orchestration/tests/test_routing_integration.py`).

### Working with the rich internal result

If you need more than `{success, department}` — confidence, evidence,
alternative routes, conflicts, etc. — call `route_envelope` (or `route`
with a `SharedStateInput`/plain dict) to get the full `RoutingResult`:

```python
result = agent.route_envelope(envelope)
print(result.confidence)          # "HIGH" / "MEDIUM" / "LOW"
print(result.routing_evidence)    # ["keyword match: ...", "strong textual relevance ..."]
print(result.alternative_routes)  # next-best candidates considered but not selected
```

`RoutingResult.as_dict()` gives the full result as a plain dict;
`RoutingResult.as_contract_dict()` gives just the `{success, department}`
pair used by `process`.

## Configuration

Tunable without touching pipeline logic — see `config.py`:

- `SCORING_WEIGHTS` — per-signal weight in the final weighted sum
- `ENABLE_LLM_SCORING` — off by default (deterministic, fully offline)
- `TOP_K_CANDIDATES`, `MIN_PRELIMINARY_SCORE` — candidate retrieval floor
- `AMBIGUITY_SCORE_MARGIN`, `CONFIDENCE_*_THRESHOLD` — decision thresholds
- `MAX_ALTERNATIVE_ROUTES`, `MAX_INTENTS` — output size caps

To route against real KutupAI department data, replace
`_SEED_DEPARTMENTS` in `knowledge_base.py` — nothing else needs to change.

## Running the tests

```bash
cd your_project   # the directory containing Agents/
python -m unittest discover -s . -p "test_*.py" -v
```

Or target a specific suite:

```bash
python -m unittest Agents.routing_agent.tests.test_routing_agent -v
python -m unittest Agents.routing_agent.tests.test_envelope_contract -v
```

Two test modules are included:

- `test_routing_agent.py` — scenario tests (normal routing, ambiguous,
  multi-intent/multi-route, conflict, missing information, adversarial
  cases) plus a metrics harness (Top-1/Top-3 accuracy, MRR, confidence
  calibration) run against a small labeled synthetic set.
- `test_envelope_contract.py` — verifies the pipeline envelope contract
  itself: output shape, passthrough of untouched stages, correct routing
  from realistic envelope data, and graceful (non-crashing) handling of
  empty/missing upstream data.

## Project layout

| File                  | Responsibility                                              |
|------------------------|--------------------------------------------------------------|
| `agent.py`             | `RoutingAgent` orchestrator; envelope-contract entry points (`process`, `route_envelope`) |
| `models.py`            | Data contracts: `SharedStateInput` (incl. `from_envelope`), `Department`, `Route`, `RoutingResult` (incl. `as_contract_dict`) |
| `knowledge_base.py`    | Department seed data + BM25 index                            |
| `tools.py`             | Individual scoring signal implementations (BM25, keyword, rule, metadata, legal, entity, semantic, LLM-pluggable) |
| `validators.py`        | Pre-routing state validation + post-routing consistency checks |
| `config.py`            | Weights, thresholds, feature toggles                         |
| `prompts.py`           | Prompt templates for the optional LLM-in-the-loop hooks      |
| `tests/`                | Scenario tests, metrics harness, envelope contract tests     |