"""
config.py
=========

Central place for scoring weights, thresholds and feature toggles so the
routing behavior can be tuned without touching pipeline logic.
"""

# Weight of each scoring signal in the final weighted sum. Renormalized at
# runtime over whichever signals are actually enabled (e.g. if the LLM
# scorer is disabled, its weight is redistributed proportionally).
SCORING_WEIGHTS = {
    "rule": 0.20,
    "keyword": 0.10,
    "bm25": 0.15,
    "semantic": 0.20,
    "metadata": 0.10,
    "legal": 0.10,
    "entity": 0.10,
    "llm": 0.05,
}

# Whether the (pluggable) LLM reasoning scorer is consulted. Off by default
# because the module must run fully offline/deterministically; wire a real
# Claude call by passing an llm_scorer into RoutingAgent (see tools.py /
# prompts.py).
ENABLE_LLM_SCORING = False

# Whether a real embedding-based semantic scorer is expected. Off by
# default; falls back to a dependency-free token-overlap/cosine scorer.
ENABLE_EXTERNAL_EMBEDDINGS = False

# --- Candidate retrieval -------------------------------------------------
TOP_K_CANDIDATES = 5
MIN_PRELIMINARY_SCORE = 0.02  # floor to even be considered a candidate

# --- Multi-intent / multi-route ------------------------------------------
MAX_INTENTS = 3
SECONDARY_ROUTE_MIN_SCORE = 0.35  # a secondary intent's best dept must clear this

# --- Ambiguity ------------------------------------------------------------
AMBIGUITY_SCORE_MARGIN = 0.06   # top1 - top2 smaller than this => ambiguous

# --- Confidence -------------------------------------------------------------
CONFIDENCE_HIGH_THRESHOLD = 0.72
CONFIDENCE_MEDIUM_THRESHOLD = 0.48
CONFLICT_PENALTY_PER_CONFLICT = 0.18
AMBIGUITY_PENALTY_PER_ITEM = 0.10
MISSING_INFO_PENALTY_PER_ITEM = 0.03

# --- Negative routing / alternatives ---------------------------------------
MAX_ALTERNATIVE_ROUTES = 2

# --- Rule scoring -----------------------------------------------------------
EXCLUDED_TOPIC_PENALTY_MULTIPLIER = 0.1  # hard-shrink score if excluded topic hit
