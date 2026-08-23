"""
tools.py
========

Scoring and retrieval primitives used by the pipeline in agent.py. Nothing
here makes a routing decision by itself -- these are individual signal
producers that agent.py combines into a hybrid score.

Includes:
    * tokenize / stopword filtering
    * BM25Index                     - classic BM25 over department corpora
    * keyword_score                 - simple overlap scoring
    * rule_score                    - routing_rules / excluded_topics logic
    * metadata_score                - sender/recipient/institution alignment
    * legal_score                   - legal_references vs legal_authority
    * entity_match_score            - entity overlap
    * SemanticScorer (ABC)          - pluggable semantic similarity
    * TokenOverlapSemanticScorer    - dependency-free default implementation
    * LLMReasoningScorer (ABC)      - pluggable LLM-in-the-loop scorer
    * HeuristicLLMScorer            - offline stand-in (documented as such)
    * retrieve_candidates           - cheap shortlist before deep scoring
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .models import Department

_STOPWORDS = {
    # Turkish
    "ve", "veya", "ile", "bu", "bir", "de", "da", "için", "olan", "olarak",
    "gibi", "ise", "ancak", "fakat", "çok", "daha", "en", "her", "tüm",
    "mi", "mu", "mı", "mü", "ki", "ne", "ya", "hem",
    # English
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is",
    "are", "with", "this", "that", "be", "as", "by", "at",
}

_TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", re.UNICODE)


def tokenize(text: Optional[str]) -> List[str]:
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _norm(text: Optional[str]) -> str:
    return (text or "").lower().strip()


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25Index:
    """Minimal, dependency-free BM25 index over a fixed set of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = [tokenize(d) for d in documents]
        self.doc_len = [len(d) for d in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_tokens else 0.0
        self.n_docs = len(self.doc_tokens)

        df: Counter = Counter()
        self.term_freqs: List[Counter] = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            self.term_freqs.append(tf)
            for term in tf:
                df[term] += 1
        self.idf: Dict[str, float] = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        if self.n_docs == 0 or self.avg_len == 0:
            return 0.0
        q_tokens = tokenize(query)
        tf = self.term_freqs[doc_index]
        dl = self.doc_len[doc_index]
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            score += idf * (freq * (self.k1 + 1)) / (denom if denom else 1)
        return score

    def max_possible_score(self, query: str) -> float:
        """Rough normalizer: score of a doc containing every query term once,
        at average length, used to squash BM25 into ~[0, 1]."""
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return 1.0
        total = 0.0
        for term in q_tokens:
            idf = self.idf.get(term, math.log(1 + (self.n_docs + 0.5) / 0.5))
            denom = 1 + self.k1 * (1 - self.b + self.b)
            total += idf * (1 * (self.k1 + 1)) / denom
        return max(total, 1e-6)

    def normalized_score(self, query: str, doc_index: int) -> float:
        raw = self.score(query, doc_index)
        cap = self.max_possible_score(query)
        return max(0.0, min(1.0, raw / cap))


# --------------------------------------------------------------------------
# Simple signal scorers
# --------------------------------------------------------------------------

def _phrase_matches(phrase: str, norm_text: str, text_tokens: set) -> bool:
    """A multi-word phrase matches only if it appears verbatim as a
    substring, or if ALL of its (non-stopword) tokens are present in the
    text -- a single shared common word (e.g. 'süreç'/'process') is not
    enough to count as a match. This avoids false positives on short,
    generic words shared across many phrases."""
    phrase_norm = _norm(phrase)
    if not phrase_norm:
        return False
    if phrase_norm in norm_text:
        return True
    phrase_tokens = set(tokenize(phrase_norm))
    if not phrase_tokens:
        return False
    return phrase_tokens.issubset(text_tokens)


def keyword_score(text: str, keywords: Sequence[str]) -> Tuple[float, List[str]]:
    """Fraction of department keywords/topics that appear in the text.
    Returns (score, matched_keywords)."""
    if not keywords:
        return 0.0, []
    norm_text = _norm(text)
    text_tokens = set(tokenize(text))
    matched = [kw for kw in keywords if _phrase_matches(kw, norm_text, text_tokens)]
    return (len(matched) / len(keywords)), matched


def rule_score(text: str, topic: Optional[str], dept: Department) -> Tuple[float, List[str], List[str], bool, Optional[str]]:
    """Evaluates routing_rules (positive signals) and excluded_topics (hard
    negative signals). Returns:
        (score, evidence_for, evidence_against, hard_excluded, exclusion_reason)
    """
    haystack = _norm(" ".join(filter(None, [text, topic])))
    haystack_tokens = set(tokenize(haystack))
    evidence_for: List[str] = []
    evidence_against: List[str] = []

    hits = 0
    for rule in dept.routing_rules:
        if _phrase_matches(rule, haystack, haystack_tokens):
            hits += 1
            evidence_for.append(f"routing rule matched: '{rule}'")
    rule_component = (hits / len(dept.routing_rules)) if dept.routing_rules else 0.0

    hard_excluded = False
    exclusion_reason = None
    for excl in dept.excluded_topics:
        if _phrase_matches(excl, haystack, haystack_tokens):
            hard_excluded = True
            exclusion_reason = f"document matches excluded topic '{excl}' for this department"
            evidence_against.append(exclusion_reason)
            break

    return rule_component, evidence_for, evidence_against, hard_excluded, exclusion_reason


def metadata_score(state, dept: Department) -> Tuple[float, List[str]]:
    evidence: List[str] = []
    points = 0.0
    total = 0.0

    total += 1.0
    if state.institution and _norm(state.institution) in _norm(dept.institution):
        points += 1.0
        evidence.append(f"institution '{state.institution}' matches '{dept.institution}'")

    total += 1.0
    recipient = _norm(state.recipient)
    if recipient and (
        recipient in _norm(dept.department)
        or recipient in _norm(dept.unit or "")
        or recipient in _norm(dept.makam or "")
    ):
        points += 1.0
        evidence.append(f"recipient '{state.recipient}' matches department/unit/makam")

    total += 1.0
    if state.previous_correspondence:
        for corr in state.previous_correspondence:
            corr_dept = _norm(str(corr.get("department", ""))) if isinstance(corr, dict) else ""
            if corr_dept and corr_dept in _norm(dept.department):
                points += 1.0
                evidence.append("previous correspondence with this department found")
                break

    return (points / total if total else 0.0), evidence


def legal_score(legal_references: Sequence[str], dept_legal_authority: Sequence[str]) -> Tuple[float, List[str]]:
    if not legal_references or not dept_legal_authority:
        return 0.0, []
    matched = []
    dept_norm = [_norm(x) for x in dept_legal_authority]
    for ref in legal_references:
        ref_norm = _norm(ref)
        if any(ref_norm in d or d in ref_norm for d in dept_norm):
            matched.append(ref)
    if not matched:
        return 0.0, []
    return (len(matched) / len(legal_references)), [f"legal reference '{m}' aligns with department authority" for m in matched]


def entity_match_score(state_entities: Sequence[str], dept_entities: Sequence[str]) -> Tuple[float, List[str]]:
    if not state_entities or not dept_entities:
        return 0.0, []
    state_norm = {_norm(e) for e in state_entities}
    dept_norm = {_norm(e) for e in dept_entities}
    matched = state_norm & dept_norm
    if not matched:
        # fall back to substring matching for partial entity names
        matched = {s for s in state_norm for d in dept_norm if s and (s in d or d in s)}
    if not matched:
        return 0.0, []
    score = len(matched) / max(len(state_norm), 1)
    return min(score, 1.0), [f"entity '{m}' matched" for m in matched]


# --------------------------------------------------------------------------
# Pluggable semantic scorer
# --------------------------------------------------------------------------

class SemanticScorer(ABC):
    """Interface for semantic similarity between a query text and a
    department's descriptive corpus. Swap in a real embedding-backed
    implementation (e.g. calling an embeddings API) without touching
    agent.py."""

    @abstractmethod
    def score(self, query_text: str, doc_text: str) -> float:
        ...


class TokenOverlapSemanticScorer(SemanticScorer):
    """Dependency-free fallback: cosine similarity over term-frequency
    vectors. Deterministic, fast, works fully offline. Intended to be
    replaced by a real embedding model in production; kept as the default
    so the agent has no hard external dependency."""

    def score(self, query_text: str, doc_text: str) -> float:
        q_tokens = tokenize(query_text)
        d_tokens = tokenize(doc_text)
        if not q_tokens or not d_tokens:
            return 0.0
        q_tf = Counter(q_tokens)
        d_tf = Counter(d_tokens)
        shared = set(q_tf) & set(d_tf)
        dot = sum(q_tf[t] * d_tf[t] for t in shared)
        q_norm = math.sqrt(sum(v * v for v in q_tf.values()))
        d_norm = math.sqrt(sum(v * v for v in d_tf.values()))
        if q_norm == 0 or d_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)


# --------------------------------------------------------------------------
# Pluggable LLM reasoning scorer
# --------------------------------------------------------------------------

class LLMReasoningScorer(ABC):
    """Interface for an (optional) LLM-in-the-loop scoring signal. The
    contract: given the state text and a department, return a (score,
    short_reason) pair. Implementations should return ONLY the final
    judgement -- no hidden chain-of-thought is ever surfaced by the agent.
    See prompts.py for a ready-to-use prompt template to wire a real
    Claude call here."""

    @abstractmethod
    def score(self, query_text: str, dept: Department) -> Tuple[float, str]:
        ...


class HeuristicLLMScorer(LLMReasoningScorer):
    """Offline stand-in used when ENABLE_LLM_SCORING is False or no LLM
    client is wired in. NOT a real LLM call -- a keyword-density heuristic
    kept structurally identical to what an LLM call would return, so it can
    be swapped for a real model without changing agent.py. Do not mistake
    this for actual LLM reasoning in production."""

    def score(self, query_text: str, dept: Department) -> Tuple[float, str]:
        score, matched = keyword_score(query_text, dept.responsibilities)
        reason = (
            f"heuristic match against responsibilities: {', '.join(matched[:3])}"
            if matched else "no strong heuristic match against responsibilities"
        )
        return score, reason


class CallableLLMScorer(LLMReasoningScorer):
    """Wraps a user-supplied callable(query_text, dept) -> (score, reason),
    e.g. a function that calls the real Anthropic API using the prompt
    template in prompts.py and parses a strict-JSON response."""

    def __init__(self, fn):
        self._fn = fn

    def score(self, query_text: str, dept: Department) -> Tuple[float, str]:
        return self._fn(query_text, dept)


# --------------------------------------------------------------------------
# Candidate retrieval
# --------------------------------------------------------------------------

def retrieve_candidates(
    query_text: str,
    departments: Sequence[Department],
    bm25_index: "BM25Index",
    dept_index_map: Dict[str, int],
    top_k: int = config.TOP_K_CANDIDATES,
) -> List[Department]:
    """Cheap first-pass retrieval (keyword + BM25 only) to shortlist
    candidates before the full multi-signal scoring pass. This keeps the
    expensive/semantic/LLM scoring bounded to top_k departments instead of
    the entire knowledge base."""

    scored = []
    for dept in departments:
        kw_score, _ = keyword_score(query_text, dept.keywords + dept.handled_topics)
        idx = dept_index_map[dept.id]
        bm25_norm = bm25_index.normalized_score(query_text, idx)
        prelim = 0.5 * kw_score + 0.5 * bm25_norm
        if prelim >= config.MIN_PRELIMINARY_SCORE:
            scored.append((prelim, dept))

    if not scored:
        # Nothing cleared the floor: still return the best few so the
        # pipeline can report low-confidence / ambiguous rather than crash.
        scored = [
            (
                0.5 * keyword_score(query_text, d.keywords + d.handled_topics)[0]
                + 0.5 * bm25_index.normalized_score(query_text, dept_index_map[d.id]),
                d,
            )
            for d in departments
        ]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]
