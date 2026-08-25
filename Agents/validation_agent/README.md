# Validation Agent

## 1. Overview

The Validation Agent is the fourth stage of the KUTUP AI document-processing
pipeline. It sits immediately after Extraction and inspects the results
produced by OCR, Classification, and Extraction to determine whether the
document has been processed reliably enough to continue toward RAG,
Summary, Routing, and Writing.

**Why this agent exists:** OCR, Classification, and Extraction are each
optimized for their own task — reading text, naming the document type,
pulling out structured fields — and none of them is responsible for
judging the *overall* trustworthiness of the pipeline's results so far.
Without a dedicated checkpoint, a document with failed extraction, a
misread field, or a low-confidence classification would flow silently
into RAG/Summary/Routing/Writing, where such problems are much harder to
detect or attribute. The Validation Agent exists to make that judgment
explicit, in one place, before downstream agents build on unreliable
data.

**Responsibility compared to the other agents:**

| Agent | Responsibility |
|---|---|
| OCR | Extracts raw text and visual signals (signatures, stamps) from the document image/PDF. |
| Classification | Assigns a `document_type` and a confidence score. |
| Extraction | Pulls structured fields (`sender`, `date`, `address`, `phone`, `email`) out of the OCR text. |
| **Validation** | **Judges whether OCR/Classification/Extraction results are complete and well-formed enough to trust, without re-doing any of their work.** |
| RAG / Summary / Routing / Writing | Consume the (now validated) document data to answer questions, summarize, route to a department, and produce final output. |

The Validation Agent never re-extracts, re-classifies, or re-reads the
document. It only inspects what earlier agents already produced.

## 2. Pipeline Position

```
Request → OCR → Classification → Extraction → Validation → RAG → Summary → Routing → Writing
```

Each agent receives the full shared state, updates only its own
namespace, and passes the whole state forward. The Validation Agent is
positioned right after Extraction and right before RAG, meaning it is
the last checkpoint before the document's data starts being used for
answering questions and producing outputs.

## 3. Input Contract

The Validation Agent receives the complete shared state dictionary, as
defined in `Validate-contract.md`:

```json
{
  "request": { "success": true, "question": "...", "document": {...} },
  "ocr": { "success": true, "ocr_data": { "page_count": 1, "language": "tr", "pages": [], "full_text": "...", "vision": {...} } },
  "classification": { "success": true, "document_type": "...", "classification_confidence": 0.95 },
  "extraction": { "success": true, "sender": null, "date": null, "address": null, "phone": null, "email": null },
  "validation": {},
  "rag": {},
  "summary": {},
  "routing": {},
  "writing": {}
}
```

| Key | Read by Validation Agent? | Purpose |
|---|---|---|
| `request` | No | Original question and document metadata; not inspected by this agent. |
| `ocr` | Yes | `ocr.success` and `ocr.ocr_data.full_text` are checked. |
| `classification` | Yes | `classification.success` and `classification.classification_confidence` are checked. |
| `extraction` | Yes | `extraction.success` and the five extracted fields (`sender`, `date`, `address`, `phone`, `email`) are checked. |
| `validation` | Overwritten | This is the only namespace the agent writes to. |
| `rag`, `summary`, `routing`, `writing` | No | Not yet populated at this pipeline stage; passed through untouched. |

The agent reads only the namespaces listed above and writes only to
`validation`. All other keys are returned exactly as received.

## 4. Output Contract

Per `Validate-contract.md`, the agent must set `state["validation"]` to:

```json
{
  "success": true,
  "is_complete": false,
  "errors": [],
  "warnings": []
}
```

| Field | Type | Meaning |
|---|---|---|
| `success` | `bool` | `True` only if `errors` is empty. Reflects whether the document can be considered validly processed. |
| `is_complete` | `bool` | `True` only if extraction succeeded **and** every one of the five extraction fields (`sender`, `date`, `address`, `phone`, `email`) is present (non-null, non-empty) **and** no hard errors were found. |
| `errors` | `list[str]` | Short string codes for problems severe enough to make the document invalid (e.g. `"extraction_failed"`, `"invalid_email_format"`). Empty when nothing is wrong. |
| `warnings` | `list[str]` | Short string codes for soft signals that don't invalidate the document but are worth surfacing (e.g. `"low_classification_confidence"`, `"ocr_failed"`). |

**This output is computed dynamically, not hardcoded.** The agent
contains no literal `{"success": True, "is_complete": False}` return
value anywhere in its logic — every field above is derived at call time
from the actual contents of `extraction`, `classification`, and `ocr` in
the incoming state. Section 6 demonstrates this with concrete examples
that each produce a different result from the same code path.

## 5. Validation Logic

All rules live in `agent.py`'s `run()` method, using pure helper
functions from `tools.py` and thresholds/patterns from `config.py`.

### Extraction validation
- **`extraction.success == False`** → hard error `"extraction_failed"`. `is_complete` is forced to `False`.
- **`extraction.success` missing/`None`** (e.g. extraction hasn't run, or state is malformed) → warning `"extraction_result_missing"`, not a hard error, since this may simply reflect pipeline ordering rather than a real failure.
- **`extraction.success == True`, some but not all of the five fields present** → warning `"partial_extraction_data"`.
- **`extraction.success == True`, all five fields null/empty** → no error, no warning by itself. This matches the canonical example in `Validate-contract.md`, where a fully-null extraction result still yields `errors: []`, `warnings: []`. Only `is_complete` reflects the missing data (set to `False`).
- **`extraction.success == True`, all five fields present** → `is_complete` is set to `True` (assuming no other errors were found).

### OCR validation
- **`ocr.success == False`** → warning `"ocr_failed"` (never a hard error).
- **`ocr.success == True` but `ocr.ocr_data.full_text` is missing or blank** → warning `"empty_ocr_text"`.

### Classification validation
- **`classification.success == False`** → warning `"classification_failed"`.
- **`classification.classification_confidence` is present and below `config.MIN_CLASSIFICATION_CONFIDENCE` (0.5)** → warning `"low_classification_confidence"`.

Classification and OCR signals are always warnings, never errors — a
weak classification or an OCR hiccup doesn't by itself mean the
document is invalid, only that it deserves a closer look downstream.

### Format validation (only when the field is present)
- **`date`**: checked with `tools.validate_date_format`, which tries `%d.%m.%Y`, `%Y-%m-%d`, and `%d/%m/%Y` via `tools.parse_date`. Invalid → error `"invalid_date_format"`.
- **`email`**: checked with `tools.validate_email_format` against `config.EMAIL_PATTERN` (`^[^@\s]+@[^@\s]+\.[^@\s]+$`). Invalid → error `"invalid_email_format"`.
- **`phone`**: checked with `tools.validate_phone_format` against `config.PHONE_DIGIT_LENGTH` (10). The function strips non-digit characters and normalizes away a leading `+90`/`90` country code or a leading trunk `0` before comparing digit count. Invalid → error `"invalid_phone_format"`.

None of these three checks run when the corresponding field is `null`
or empty — they are never treated as required.

### Failure isolation
The entire `run()` body is wrapped in a `try/except`. If anything
raises unexpectedly, the agent returns:
```json
{ "success": false, "is_complete": false, "errors": ["validation_agent_internal_error:<message>"], "warnings": [] }
```
instead of crashing the pipeline, while still respecting the exact
output schema.

## 6. Dynamic Behavior Examples

These are real outputs from running `ValidationAgent().run(state)` with
different inputs, not illustrative pseudocode:

**Example 1 — Full extraction data → `is_complete: true`**
```json
// extraction: {"success": true, "sender": "Ali Veli", "date": "01.03.2026",
//              "address": "Ankara", "phone": "5551234567", "email": "ali@example.com"}
{ "success": true, "is_complete": true, "errors": [], "warnings": [] }
```

**Example 2 — Failed extraction → error generated**
```json
// extraction: {"success": false, ...all fields null...}
{ "success": false, "is_complete": false, "errors": ["extraction_failed"], "warnings": [] }
```

**Example 3 — Low classification confidence → warning generated**
```json
// classification: {"success": true, "classification_confidence": 0.15, ...}
// (combined with partial extraction and OCR failure in the same run)
{
  "success": false,
  "is_complete": false,
  "errors": ["invalid_email_format"],
  "warnings": ["partial_extraction_data", "low_classification_confidence", "ocr_failed"]
}
```

**Example 4 — OCR failure → warning generated**
```json
// ocr: {"success": false, "ocr_data": {}}
// (warning shown above alongside other signals: "ocr_failed" appears
//  in the warnings list regardless of what else is happening)
```

**Contract canonical example** (all extraction fields null, everything
else successful) reproduces exactly the sample from `Validate-contract.md`:
```json
{ "success": true, "is_complete": false, "errors": [], "warnings": [] }
```

The same code path produces four structurally different results above
because every value is computed from the specific input, confirming
there is no hardcoded response anywhere in the implementation.

## 7. Shared State Preservation

The Validation Agent reads from `state["extraction"]`,
`state["classification"]`, and `state["ocr"]`, and writes to exactly one
key: `state["validation"]`. It never modifies, deletes, or reorders:

- `request`
- `ocr`
- `classification`
- `extraction`
- `rag`
- `summary`
- `routing`
- `writing`

This is verified directly by `test_shared_state_preserved`, which
snapshots `request`, `ocr`, `classification`, and `extraction` before
calling `run()` and asserts they are byte-for-byte identical afterward,
and that `rag`, `summary`, `routing`, and `writing` remain untouched
empty dicts.

## 8. Testing

`Tests/Agents/test_validation_agent.py` contains **23 tests**, all
passing (`23 passed`) when run with:

```powershell
python -m pytest Tests\Agents\test_validation_agent.py -v
```

Test categories:

| Category | Count | Examples |
|---|---|---|
| Contract compatibility | 3 | Exact reproduction of the `Validate-contract.md` canonical input/output pair; state preservation; compatibility with what Routing expects to see already populated. |
| Schema validation | 1 | Confirms the output has exactly the four contract keys (`success`, `is_complete`, `errors`, `warnings`) and no leftover fields from earlier design iterations (e.g. no `status`, `confidence`, `checked_rules`). |
| Extraction | 4 | Full success marks complete; hard failure produces an error; partial data produces a warning; fully-null data produces neither, matching the contract example. |
| OCR | 2 | OCR failure produces a warning; empty OCR text produces a warning. |
| Classification | 2 | Classification failure produces a warning; low confidence produces a warning. |
| Format validation | 8 | Invalid/valid date, email, and phone checks at both the agent level (only checked when the field is present) and the `tools.py` unit level. |
| Edge cases | 3 | Agent registration under the correct name; a missing `extraction` key doesn't crash the agent; a completely empty state dict doesn't crash the agent. |

## 9. File Structure

```
validation_agent/
│
├── agent.py       # ValidationAgent class: reads state, computes validation, writes state["validation"]
├── tools.py       # Pure, stateless helper functions (date/email/phone format checks)
├── config.py      # Thresholds and patterns (MIN_CLASSIFICATION_CONFIDENCE, EMAIL_PATTERN, PHONE_DIGIT_LENGTH)
├── prompts.py     # Reserved for future semantic/LLM prompts; currently unused - no LLM client is imported
├── __init__.py
└── README.md      # This file
```

## 10. Design Decisions

**Why is Validation a separate agent instead of being folded into
Extraction or Classification?**

- **Modular architecture.** Extraction's job is to pull fields out of
  OCR text; Classification's job is to name the document type. Neither
  is naturally responsible for judging whether the *overall* pipeline
  result is trustworthy. Keeping that judgment in its own agent means
  each agent has one clear responsibility.
- **Easier debugging.** When something goes wrong downstream (e.g. RAG
  answers a question incorrectly), it's immediately possible to check
  `state["validation"]` and see exactly which upstream signal was
  flagged, rather than having to infer it from Extraction's or
  Classification's internal logic.
- **Independent testing.** Because Validation only depends on the
  *shape* of `extraction`, `classification`, and `ocr` output — not on
  how those agents produce that output internally — its 23 tests can
  run in complete isolation, using plain dictionaries that mimic the
  contract, with no OCR engine, classifier model, or extraction logic
  involved at all.
- **Safer pipeline.** A dedicated checkpoint means every document
  passes through the same validation logic regardless of which
  document type or extraction path it took, rather than each agent
  needing to duplicate its own validation rules.

  Testing 
  python -m pytest Tests\Agents\test_validation_agent.py::test_valid_pipeline_state_matches_contract_example -v -s