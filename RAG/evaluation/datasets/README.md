# Evaluation datasets

`eval_set.json` is an automatically generated regression dataset. It is useful
for detecting accidental retrieval regressions, but it must not be used as the
sole evidence that a retrieval strategy is better: its questions are derived
from the indexed passages.

Before changing production retrieval settings, create a separate JSON dataset
with `"split": "heldout"`. Write questions independently from the corpus,
review their expected law/article IDs with a legal-domain expert, and include
both ambiguity and near-miss cases. Run it with:

```bash
python -m RAG.evaluation.benchmark --dataset path/to/heldout.json --split heldout
```

Each example needs a `query` plus either exact `expected_law_numbers` and
`expected_article_numbers` (preferred for legal questions),
`relevant_chunk_ids`, or `relevant_source_files`. Do not include answers or
source snippets in the query itself.
## Dataset types

- `eval_set.json`: automatically generated synthetic regression data. It is
  useful for detecting a code regression, but must not be presented as a
  measure of real legal-question quality.
- `heldout_legal.json`: human-authored Turkish legal paraphrases and short
  scenarios. Each record is labelled with an expected law and article and is
  used by `RAG.evaluation.system_benchmark` for an exact-citation score.

Before publishing a score externally, have a qualified legal reviewer confirm
the cited current-law article for every held-out record and record the corpus
version used for the test.
