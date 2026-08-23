"""
agent.py
========

RoutingAgent: orchestrates the full pipeline described in the specification.

    Shared State
        -> State Validation
        -> Document / Intent Analysis
        -> Entity & Topic Extraction
        -> Multi-Intent Detection
        -> Candidate Department Retrieval
        -> Rule + Semantic + Metadata Scoring   (+ BM25, keyword, legal, entity, LLM)
        -> Hierarchy Resolution
        -> Authority Resolution
        -> Conflict Detection
        -> Ambiguity Detection
        -> Routing Decision
        -> Routing Validation
        -> Final Route + Reason + Evidence + Confidence

Routing is never a single LLM call: it is a weighted combination of eight
independent signals (rule, keyword, BM25, semantic, metadata, legal,
entity, optional-LLM), followed by deterministic hierarchy/authority
resolution and validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from . import config
from .knowledge_base import KnowledgeBase, default_knowledge_base
from .models import (
    ConfidenceLevel,
    Department,
    IntentSegment,
    ProcessingChannel,
    Route,
    RouteCandidate,
    RoutingResult,
    RoutingStatus,
    ScoreBreakdown,
    SharedStateInput,
)
from . import tools
from . import validators


class RoutingAgent:
    """Hybrid, explainable, hierarchy-aware document routing agent."""

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        semantic_scorer: Optional[tools.SemanticScorer] = None,
        llm_scorer: Optional[tools.LLMReasoningScorer] = None,
        enable_llm_scoring: Optional[bool] = None,
    ):
        self.kb = knowledge_base or default_knowledge_base()
        self.semantic_scorer = semantic_scorer or tools.TokenOverlapSemanticScorer()
        self.enable_llm_scoring = (
            enable_llm_scoring if enable_llm_scoring is not None else config.ENABLE_LLM_SCORING
        )
        self.llm_scorer = llm_scorer or tools.HeuristicLLMScorer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def route(self, shared_state: Union[SharedStateInput, Dict[str, Any]]) -> RoutingResult:
        state = self._coerce_state(shared_state)

        # 1. State validation
        missing_information = validators.validate_shared_state(state)

        # 2 & 3. Document/intent analysis + entity & topic extraction are
        # largely pre-computed upstream; we consume what's given and derive
        # a normalized "analysis text" used throughout scoring.
        analysis_text = self._build_analysis_text(state)

        # 4. Multi-intent detection
        intents = self._detect_intents(state, analysis_text)
        is_multi = len(intents) > 1

        # 5-6. Candidate retrieval + hybrid scoring, per intent
        ranked_by_intent: List[Tuple[IntentSegment, List[RouteCandidate]]] = []
        for intent in intents:
            candidates = self._retrieve_candidates(intent.text)
            scored = self._score_candidates(state, intent, candidates)
            scored.sort(key=lambda c: c.score.total, reverse=True)
            ranked_by_intent.append((intent, scored))

        primary_intent, primary_ranked = ranked_by_intent[0]
        if not primary_ranked:
            return self._empty_result(missing_information)

        primary_candidate = primary_ranked[0]

        # 9. Conflict detection (upstream-signal consistency)
        conflicts = self._detect_conflicts(state, primary_candidate)

        # 10. Ambiguity detection
        ambiguities = self._detect_ambiguity(primary_ranked)

        # 7 & 8. Hierarchy + authority resolution happen while building Route
        channel = self._determine_channel(state, primary_candidate)
        primary_route = self._build_route(primary_candidate, channel)

        # Secondary routes from additional intents (multi-route support)
        secondary_routes: List[Route] = []
        if is_multi:
            for intent, ranked in ranked_by_intent[1:]:
                if not ranked:
                    continue
                top = ranked[0]
                if top.score.total >= config.SECONDARY_ROUTE_MIN_SCORE and not top.rejected:
                    if top.department.department != primary_candidate.department.department:
                        sec_channel = self._determine_channel(state, top)
                        secondary_routes.append(self._build_route(top, sec_channel))

        if not secondary_routes:
            is_multi = False

        # Alternative / negative routing: next-best distinct candidates
        alternative_routes = self._build_alternative_routes(primary_ranked, primary_candidate)

        # 11. Routing decision -> status
        routing_status = self._determine_status(is_multi, conflicts, ambiguities)

        # 12. Confidence calculation
        confidence_level, confidence_score = self._calculate_confidence(
            primary_candidate, conflicts, ambiguities, missing_information
        )

        needs_human_review = (
            confidence_level == ConfidenceLevel.LOW.value
            or bool(conflicts)
            or bool(ambiguities)
        )

        routing_reason = self._compose_reason(primary_candidate)
        routing_evidence = list(primary_candidate.evidence_for)

        result = RoutingResult(
            routing_status=routing_status,
            primary_route=primary_route,
            secondary_routes=secondary_routes,
            alternative_routes=alternative_routes,
            recommended_department=primary_route.department,
            recommended_authority=primary_route.authority,
            routing_reason=routing_reason,
            routing_evidence=routing_evidence,
            confidence=confidence_level,
            ambiguities=ambiguities,
            conflicts=conflicts,
            missing_information=missing_information,
            needs_human_review=needs_human_review,
        )

        # 13. Routing validation
        validation_problems = validators.validate_routing_result(result, self.kb)
        if validation_problems:
            result.conflicts = result.conflicts + [f"ROUTING_CONFLICT: {p}" for p in validation_problems]
            result.routing_status = RoutingStatus.ROUTING_CONFLICT.value
            result.needs_human_review = True
            # Recompute confidence with the validation penalty applied.
            downgraded_score = max(0.0, confidence_score - len(validation_problems) * 0.1)
            result.confidence = self._confidence_level_from_score(downgraded_score)

        return result

    # ------------------------------------------------------------------
    # Envelope contract entry point
    # ------------------------------------------------------------------

    def route_envelope(self, envelope: Dict[str, Any]) -> RoutingResult:
        """Same as `route`, but consumes the pipeline envelope contract
        directly (see models.SharedStateInput.from_envelope) instead of a
        pre-built SharedStateInput."""
        state = SharedStateInput.from_envelope(envelope)
        return self.route(state)

    def process(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        """Full contract adapter.

        Input : the pipeline envelope
                 {request, ocr, classification, extraction, validation,
                  rag, summary, routing, writing}

        Output: the SAME envelope, unchanged except that `routing` is
                replaced with:
                    {"success": bool, "department": string}

        Prefer ``run`` when calling from Orchestration (GraphState); use
        ``process`` for the standalone pipeline envelope.
        """
        result = self.route_envelope(envelope)
        updated_envelope = dict(envelope)
        updated_envelope["routing"] = result.as_contract_dict()
        return updated_envelope

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestration entry: adapt GraphState → envelope, write ``routing``.

        Reads upstream sections in either pipeline shape (``ocr``,
        ``classification``, …) or GraphState legacy shape (``ocr_result``,
        ``classification_result``, ``rag_result``, …). Writes only:

            state["routing"] = {"success": bool, "department": str}
        """
        if not isinstance(state, dict):
            raise TypeError("RoutingAgent.run expects GraphState as a dict")

        updated = dict(state)
        routed = self.process(self._graph_state_to_envelope(updated))
        updated["routing"] = routed.get("routing") or {
            "success": False,
            "department": "",
        }
        return updated

    @staticmethod
    def _graph_state_to_envelope(state: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Orchestration GraphState into the pipeline envelope."""

        def _as_dict(value: Any) -> Dict[str, Any]:
            return value if isinstance(value, dict) else {}

        request = dict(_as_dict(state.get("request")))
        question = (
            request.get("question")
            or state.get("question")
            or state.get("accompanying_text")
            or ""
        )
        document = _as_dict(request.get("document"))
        if not document:
            document = {
                "document_id": state.get("document_id") or request.get("document_id") or "",
                "file_name": request.get("file_name") or "",
                "file_type": request.get("file_type") or "",
            }
        request = {
            **request,
            "success": request.get("success", True),
            "question": question,
            "document": document,
        }

        ocr = _as_dict(state.get("ocr"))
        if not _as_dict(ocr.get("ocr_data")).get("full_text"):
            ocr_result = _as_dict(state.get("ocr_result"))
            full_text = ""
            data = ocr_result.get("Data") or ocr_result.get("data") or []
            if isinstance(data, list) and data and isinstance(data[0], dict):
                full_text = data[0].get("full_text") or ""
            full_text = (
                full_text
                or state.get("document_text")
                or state.get("text")
                or ""
            )
            ocr = {
                "success": bool(full_text) or bool(ocr_result.get("Success")),
                "ocr_data": {
                    "page_count": _as_dict(ocr.get("ocr_data")).get("page_count") or 1,
                    "language": _as_dict(ocr.get("ocr_data")).get("language") or "tr",
                    "pages": _as_dict(ocr.get("ocr_data")).get("pages") or [],
                    "full_text": full_text,
                    "vision": _as_dict(ocr.get("ocr_data")).get("vision")
                    or {
                        "signature": {"detected": False, "handwritten": False},
                        "stamp": {"detected": False},
                    },
                },
            }

        classification = _as_dict(state.get("classification"))
        if not classification.get("document_type"):
            cr = _as_dict(state.get("classification_result"))
            classification = {
                "success": True,
                "document_type": cr.get("document_type") or cr.get("doc_type") or "",
                "classification_confidence": cr.get(
                    "classification_confidence", classification.get("classification_confidence", 0.0)
                ),
            }

        extraction = _as_dict(state.get("extraction")) or _as_dict(
            state.get("extraction_result")
        )
        validation = _as_dict(state.get("validation")) or _as_dict(
            state.get("validation_result")
        )

        rag = _as_dict(state.get("rag"))
        if not _as_dict(rag.get("rag_data")):
            rag_result = _as_dict(state.get("rag_result"))
            data = _as_dict(rag_result.get("data"))
            if data or rag_result:
                rag = {
                    "success": rag_result.get("success", True),
                    "rag_data": {
                        "operation": data.get("operation", "retrieve"),
                        "query": data.get("query") or question,
                        "results": data.get("results") or [],
                    },
                }

        summary = _as_dict(state.get("summary"))

        return {
            "request": request,
            "ocr": ocr,
            "classification": classification,
            "extraction": extraction,
            "validation": validation,
            "rag": rag,
            "summary": summary,
            "routing": _as_dict(state.get("routing")),
            "writing": _as_dict(state.get("writing")),
        }

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _coerce_state(self, shared_state: Union[SharedStateInput, Dict[str, Any]]) -> SharedStateInput:
        if isinstance(shared_state, SharedStateInput):
            return shared_state
        return SharedStateInput.from_dict(dict(shared_state))

    def _build_analysis_text(self, state: SharedStateInput) -> str:
        parts = [
            state.summary,
            state.intent,
            state.requested_action,
            state.topic,
            " ".join(state.subtopics),
            state.document_text,
        ]
        return " ".join(p for p in parts if p)

    def _quick_best_department(self, text: str) -> Optional[Department]:
        best, best_score = None, 0.0
        for dept in self.kb.all():
            score, _ = tools.keyword_score(text, dept.keywords + dept.handled_topics)
            if score > best_score:
                best, best_score = dept, score
        return best if best_score > 0 else None

    def _detect_intents(self, state: SharedStateInput, analysis_text: str) -> List[IntentSegment]:
        """Heuristic multi-intent detection: the primary intent is the
        whole analysis text. A subtopic is promoted to its own secondary
        intent only when it clearly points at a *different* department
        than the primary text does -- i.e. only when evidence supports it,
        per the specification's "do not use multi-route unless evidence
        supports it"."""
        intents = [IntentSegment(label="primary", text=analysis_text, source_topic=state.topic)]

        if not state.subtopics:
            return intents

        primary_dept = self._quick_best_department(analysis_text)
        for sub in state.subtopics:
            if len(intents) >= config.MAX_INTENTS:
                break
            sub_dept = self._quick_best_department(sub)
            if sub_dept and (not primary_dept or sub_dept.department != primary_dept.department):
                intents.append(IntentSegment(label=f"secondary:{sub}", text=sub, source_topic=sub))

        return intents

    def _retrieve_candidates(self, query_text: str) -> List[Department]:
        return tools.retrieve_candidates(
            query_text,
            self.kb.all(),
            self.kb.bm25,
            self.kb.index_map(),
            top_k=config.TOP_K_CANDIDATES,
        )

    def _score_candidates(
        self, state: SharedStateInput, intent: IntentSegment, candidates: List[Department]
    ) -> List[RouteCandidate]:
        results: List[RouteCandidate] = []
        index_map = self.kb.index_map()

        # Drop signals that are structurally inapplicable given what the
        # upstream state actually provides (e.g. no legal_references were
        # extracted, or the LLM scorer is disabled), and redistribute their
        # weight proportionally over the remaining signals. This avoids
        # unfairly penalizing a department for the *absence of input data*
        # rather than a genuine mismatch.
        weights = dict(config.SCORING_WEIGHTS)
        inapplicable = []
        if not self.enable_llm_scoring:
            inapplicable.append("llm")
        if not state.legal_references:
            inapplicable.append("legal")
        if not state.entities:
            inapplicable.append("entity")
        dropped_weight = sum(weights.pop(k) for k in inapplicable)
        if weights:
            redistribute = dropped_weight / len(weights)
            weights = {k: v + redistribute for k, v in weights.items()}

        for dept in candidates:
            evidence_for: List[str] = []
            evidence_against: List[str] = []

            kw_score, kw_matched = tools.keyword_score(intent.text, dept.keywords + dept.handled_topics)
            if kw_matched:
                evidence_for.append(f"keyword match: {', '.join(kw_matched[:5])}")

            bm25_score = self.kb.bm25.normalized_score(intent.text, index_map[dept.id])
            if bm25_score > 0.15:
                evidence_for.append(f"strong textual relevance (BM25={bm25_score:.2f})")

            rule_s, rule_ev_for, rule_ev_against, hard_excluded, exclusion_reason = tools.rule_score(
                intent.text, intent.source_topic, dept
            )
            evidence_for.extend(rule_ev_for)
            evidence_against.extend(rule_ev_against)

            semantic_s = self.semantic_scorer.score(intent.text, dept.corpus_text())
            if semantic_s > 0.2:
                evidence_for.append(f"semantic similarity to department scope (score={semantic_s:.2f})")

            metadata_s, metadata_ev = tools.metadata_score(state, dept)
            evidence_for.extend(metadata_ev)

            legal_s, legal_ev = tools.legal_score(state.legal_references, dept.legal_authority)
            evidence_for.extend(legal_ev)

            entity_s, entity_ev = tools.entity_match_score(state.entities, dept.entities)
            evidence_for.extend(entity_ev)

            if self.enable_llm_scoring:
                llm_s, llm_reason = self.llm_scorer.score(intent.text, dept)
                if llm_reason:
                    evidence_for.append(f"llm assessment: {llm_reason}")
            else:
                llm_s = 0.0

            score = ScoreBreakdown(
                rule=rule_s, keyword=kw_score, bm25=bm25_score, semantic=semantic_s,
                metadata=metadata_s, legal=legal_s, entity=entity_s, llm=llm_s,
            )
            total = (
                weights.get("rule", 0.0) * rule_s
                + weights.get("keyword", 0.0) * kw_score
                + weights.get("bm25", 0.0) * bm25_score
                + weights.get("semantic", 0.0) * semantic_s
                + weights.get("metadata", 0.0) * metadata_s
                + weights.get("legal", 0.0) * legal_s
                + weights.get("entity", 0.0) * entity_s
                + weights.get("llm", 0.0) * llm_s
            )

            rejected = False
            rejection_reason = None
            if hard_excluded:
                total *= config.EXCLUDED_TOPIC_PENALTY_MULTIPLIER
                rejected = True
                rejection_reason = exclusion_reason

            if total <= 0.0 and not evidence_for:
                rejected = True
                rejection_reason = rejection_reason or "no supporting evidence found for this department"
                if not evidence_against:
                    evidence_against.append(rejection_reason)

            score.total = max(0.0, min(1.0, total))

            results.append(
                RouteCandidate(
                    department=dept,
                    score=score,
                    evidence_for=evidence_for,
                    evidence_against=evidence_against,
                    rejected=rejected,
                    rejection_reason=rejection_reason,
                    intent_label=intent.label,
                )
            )

        return results

    def _detect_conflicts(self, state: SharedStateInput, primary: RouteCandidate) -> List[str]:
        """Detects inconsistency between upstream agent signals (analysis,
        writing) and the department the routing decision is pointing at.
        e.g. document classified as personnel, but analysis/writing carries
        strong legal-objection language, yet routing points at HR."""
        conflicts: List[str] = []

        legal_dept = self.kb.get_by_name("Hukuk Müşavirliği")
        if legal_dept and primary.department.department != legal_dept.department:
            signal_text = " ".join(
                filter(None, [state.intent, state.writing_output, state.metadata.get("analysis_summary")])
            )
            if signal_text:
                legal_score, matched = tools.keyword_score(signal_text, legal_dept.keywords)
                primary_score, _ = tools.keyword_score(signal_text, primary.department.keywords)
                if legal_score >= 0.34 and legal_score > primary_score + 0.15:
                    conflicts.append(
                        "CONFLICT_DETECTED: analysis/writing signals "
                        f"({', '.join(matched)}) point to '{legal_dept.department}' "
                        f"but routing selected '{primary.department.department}'"
                    )

        classification_topic = state.metadata.get("classification_topic")
        if classification_topic and classification_topic not in primary.department.handled_topics:
            cls_dept = None
            best = 0.0
            for dept in self.kb.all():
                s, _ = tools.keyword_score(classification_topic, dept.handled_topics)
                if s > best:
                    best, cls_dept = s, dept
            if cls_dept and best >= 0.5 and cls_dept.department != primary.department.department:
                conflicts.append(
                    f"CONFLICT_DETECTED: upstream classification topic '{classification_topic}' "
                    f"suggests '{cls_dept.department}', but routing selected '{primary.department.department}'"
                )

        return conflicts

    def _detect_ambiguity(self, ranked: List[RouteCandidate]) -> List[str]:
        ambiguities: List[str] = []
        viable = [c for c in ranked if not c.rejected]
        if len(viable) >= 2:
            top1, top2 = viable[0], viable[1]
            gap = top1.score.total - top2.score.total
            if gap < config.AMBIGUITY_SCORE_MARGIN:
                ambiguities.append(
                    f"'{top1.department.department}' (score={top1.score.total:.2f}) and "
                    f"'{top2.department.department}' (score={top2.score.total:.2f}) are within "
                    f"{config.AMBIGUITY_SCORE_MARGIN} of each other"
                )
        return ambiguities

    def _determine_channel(self, state: SharedStateInput, candidate: RouteCandidate) -> str:
        text = (state.document_text or "") + " " + (state.intent or "")
        text_low = text.lower()
        if candidate.department.channel_hint == "CONFIDENTIAL":
            return ProcessingChannel.CONFIDENTIAL.value
        if candidate.department.channel_hint == "LEGAL" or state.legal_references:
            return ProcessingChannel.LEGAL.value
        if candidate.department.channel_hint == "EXTERNAL_CORRESPONDENCE":
            return ProcessingChannel.EXTERNAL_CORRESPONDENCE.value
        if any(w in text_low for w in ("acil", "ivedi", "urgent")):
            return ProcessingChannel.URGENT.value
        if state.institution and candidate.department.institution.lower() not in state.institution.lower():
            return ProcessingChannel.EXTERNAL_CORRESPONDENCE.value
        return ProcessingChannel.STANDARD.value

    def _build_route(self, candidate: RouteCandidate, channel: str) -> Route:
        dept = candidate.department
        return Route(
            institution=dept.institution,
            presidency=dept.presidency,
            department=dept.department,
            unit=dept.unit,
            branch=dept.branch,
            authority=dept.authority_level,
            makam=dept.makam,
            channel=channel,
            reason=self._compose_reason(candidate),
            evidence=list(candidate.evidence_for),
            score=candidate.score.total,
            rejection_reason=candidate.rejection_reason,
        )

    def _build_alternative_routes(
        self, ranked: List[RouteCandidate], primary: RouteCandidate
    ) -> List[Route]:
        alternatives = []
        seen = {primary.department.department}
        for cand in ranked:
            if len(alternatives) >= config.MAX_ALTERNATIVE_ROUTES:
                break
            if cand.department.department in seen:
                continue
            seen.add(cand.department.department)
            reason = cand.rejection_reason or "lower combined score than the primary route"
            route = Route(
                institution=cand.department.institution,
                presidency=cand.department.presidency,
                department=cand.department.department,
                unit=cand.department.unit,
                branch=cand.department.branch,
                authority=cand.department.authority_level,
                makam=cand.department.makam,
                channel=self._determine_channel_for_alt(cand),
                reason=f"considered but not selected: {reason}",
                evidence=cand.evidence_for[:3],
                score=cand.score.total,
                rejection_reason=reason,
            )
            alternatives.append(route)
        return alternatives

    def _determine_channel_for_alt(self, candidate: RouteCandidate) -> str:
        if candidate.department.channel_hint == "CONFIDENTIAL":
            return ProcessingChannel.CONFIDENTIAL.value
        if candidate.department.channel_hint == "LEGAL":
            return ProcessingChannel.LEGAL.value
        if candidate.department.channel_hint == "EXTERNAL_CORRESPONDENCE":
            return ProcessingChannel.EXTERNAL_CORRESPONDENCE.value
        return ProcessingChannel.STANDARD.value

    def _determine_status(self, is_multi: bool, conflicts: List[str], ambiguities: List[str]) -> str:
        if conflicts:
            return RoutingStatus.CONFLICT_DETECTED.value
        if ambiguities:
            return RoutingStatus.AMBIGUOUS.value
        if is_multi:
            return RoutingStatus.MULTI_ROUTE.value
        return RoutingStatus.SINGLE_ROUTE.value

    def _confidence_level_from_score(self, score: float) -> str:
        if score >= config.CONFIDENCE_HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH.value
        if score >= config.CONFIDENCE_MEDIUM_THRESHOLD:
            return ConfidenceLevel.MEDIUM.value
        return ConfidenceLevel.LOW.value

    def _calculate_confidence(
        self,
        primary: RouteCandidate,
        conflicts: List[str],
        ambiguities: List[str],
        missing_information: List[str],
    ) -> Tuple[str, float]:
        base = primary.score.total
        # Reward broad evidence coverage (more independent signals firing).
        active_signals = sum(
            1 for v in [primary.score.rule, primary.score.keyword, primary.score.bm25,
                        primary.score.semantic, primary.score.metadata, primary.score.legal,
                        primary.score.entity] if v > 0.05
        )
        coverage_bonus = min(0.10, 0.015 * active_signals)

        penalty = (
            config.CONFLICT_PENALTY_PER_CONFLICT * len(conflicts)
            + config.AMBIGUITY_PENALTY_PER_ITEM * len(ambiguities)
            + config.MISSING_INFO_PENALTY_PER_ITEM * len(missing_information)
        )

        adjusted = max(0.0, min(1.0, base + coverage_bonus - penalty))
        level = self._confidence_level_from_score(adjusted)
        # Conflicts or ambiguity can never yield HIGH confidence, even if
        # the raw score is high -- low confidence must not be masked.
        if (conflicts or ambiguities) and level == ConfidenceLevel.HIGH.value:
            level = ConfidenceLevel.MEDIUM.value
        return level, adjusted

    def _compose_reason(self, candidate: RouteCandidate) -> str:
        dept = candidate.department
        if candidate.evidence_for:
            top_evidence = candidate.evidence_for[0]
            return f"Routed to {dept.department} based on {top_evidence}."
        return f"Routed to {dept.department} as the best-scoring available candidate."

    def _empty_result(self, missing_information: List[str]) -> RoutingResult:
        missing_information = missing_information + ["no viable department candidates found"]
        return RoutingResult(
            routing_status=RoutingStatus.AMBIGUOUS.value,
            primary_route=None,
            secondary_routes=[],
            alternative_routes=[],
            recommended_department=None,
            recommended_authority=None,
            routing_reason="No department candidate cleared the minimum evidence threshold.",
            routing_evidence=[],
            confidence=ConfidenceLevel.LOW.value,
            ambiguities=["no candidates retrieved"],
            conflicts=[],
            missing_information=missing_information,
            needs_human_review=True,
        )


# --------------------------------------------------------------------------
# Module-level convenience API (functional wrapper over a shared default
# agent instance) for callers that just want to hand in/receive the
# envelope contract without managing a RoutingAgent themselves.
# --------------------------------------------------------------------------

_default_agent: Optional[RoutingAgent] = None


def get_default_agent() -> RoutingAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = RoutingAgent()
    return _default_agent


def process(envelope: Dict[str, Any], agent: Optional[RoutingAgent] = None) -> Dict[str, Any]:
    """Functional entry point matching the pipeline contract:

        routing input  : {request, ocr, classification, extraction,
                           validation, rag, summary, routing: {}, writing}
        routing output : same envelope with
                           routing: {"success": bool, "department": string}
    """
    return (agent or get_default_agent()).process(envelope)
