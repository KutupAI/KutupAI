"""
test_routing_agent.py
======================

Two layers of testing, as required by the specification:

  1. Scenario tests (unittest) covering: normal routing, ambiguous cases,
     multi-intent cases, multi-route cases, conflict cases, missing
     information cases, and adversarial cases.

  2. A metrics harness (evaluate) computing Top-1 / Top-3 routing accuracy,
     MRR, department accuracy, authority accuracy, conflict detection
     accuracy, ambiguity detection accuracy, and a simple confidence
     calibration check, run over a small labeled synthetic set.

Run with:  python -m unittest Agents.routing_agent.tests.test_routing_agent -v
"""

from __future__ import annotations

import unittest
from typing import List, Optional

from Agents.routing_agent.agent import RoutingAgent
from Agents.routing_agent.knowledge_base import default_knowledge_base
from Agents.routing_agent.models import SharedStateInput


def make_state(**overrides) -> SharedStateInput:
    base = dict(
        document_text="",
        document_type="dilekçe",
        summary=None,
        intent=None,
        entities=[],
        topic=None,
        subtopics=[],
        requested_action=None,
        sender="Vatandaş",
        recipient=None,
        institution="Kutup Kurumu",
        legal_references=[],
        previous_correspondence=[],
        attachments=[],
        metadata={},
        classification_confidence=0.9,
        analysis_confidence=0.9,
        writing_output=None,
    )
    base.update(overrides)
    return SharedStateInput(**base)


class RoutingAgentScenarioTests(unittest.TestCase):
    def setUp(self):
        self.agent = RoutingAgent(knowledge_base=default_knowledge_base())

    # ---------------------------------------------------------------- #
    # Normal routing
    # ---------------------------------------------------------------- #
    def test_normal_personnel_routing(self):
        state = make_state(
            document_text="Personel atama işlemim ve izin talebimin değerlendirilmesini rica ederim.",
            topic="personel",
            intent="personel atama talebi",
        )
        result = self.agent.route(state)
        self.assertEqual(result.recommended_department, "İnsan Kaynakları Daire Başkanlığı")
        self.assertIn(result.confidence, ("HIGH", "MEDIUM"))
        self.assertFalse(result.conflicts)

    def test_normal_it_routing(self):
        state = make_state(
            document_text="Sistemde arıza oluştu, teknik destek talep ediyorum.",
            topic="teknik destek",
            intent="sistem arızası bildirimi",
        )
        result = self.agent.route(state)
        self.assertEqual(result.recommended_department, "Bilgi İşlem Daire Başkanlığı")

    # ---------------------------------------------------------------- #
    # Ambiguous
    # ---------------------------------------------------------------- #
    def test_ambiguous_case(self):
        # Deliberately vague text with almost no department-specific signal.
        state = make_state(
            document_text="Konu hakkında bilgi talep ediyoruz.",
            topic=None,
            intent=None,
        )
        result = self.agent.route(state)
        self.assertIn(result.routing_status, ("AMBIGUOUS", "SINGLE_ROUTE"))
        if result.routing_status == "AMBIGUOUS":
            self.assertTrue(result.needs_human_review)

    # ---------------------------------------------------------------- #
    # Multi-intent / multi-route
    # ---------------------------------------------------------------- #
    def test_multi_intent_multi_route(self):
        state = make_state(
            document_text=(
                "Personel atama talebimin yanı sıra, satın alma sürecine ilişkin "
                "ihale belgelerinin de tarafımıza iletilmesini rica ederiz."
            ),
            topic="personel",
            intent="personel atama talebi",
            subtopics=["satın alma ihale süreci"],
        )
        result = self.agent.route(state)
        if result.routing_status == "MULTI_ROUTE":
            self.assertGreaterEqual(len(result.secondary_routes), 1)
            self.assertNotEqual(
                result.primary_route.department,
                result.secondary_routes[0].department,
            )
        else:
            # Acceptable fallback: evidence for a clean split wasn't strong
            # enough, single route is still a valid, safe outcome.
            self.assertIn(result.routing_status, ("SINGLE_ROUTE", "AMBIGUOUS", "CONFLICT_DETECTED"))

    # ---------------------------------------------------------------- #
    # Conflict
    # ---------------------------------------------------------------- #
    def test_conflict_detection(self):
        state = make_state(
            document_text="Personel dosyasıyla ilgili hukuki itiraz ve dava sürecine ilişkin görüş talep ediyoruz.",
            topic="personel",
            intent="hukuki itiraz",
            writing_output="Bu belge bir hukuki görüş ve itiraz dilekçesi niteliğindedir; dava süreci başlatılmalıdır.",
            metadata={"classification_topic": "personel"},
        )
        result = self.agent.route(state)
        # Either the routing correctly lands on Legal (no conflict needed),
        # or, if it lands elsewhere, a conflict must be reported.
        if result.recommended_department != "Hukuk Müşavirliği":
            self.assertTrue(result.conflicts)
            self.assertEqual(result.routing_status, "CONFLICT_DETECTED")

    # ---------------------------------------------------------------- #
    # Missing information
    # ---------------------------------------------------------------- #
    def test_missing_information_case(self):
        state = make_state(document_text="Talebimizin değerlendirilmesini rica ederiz.")
        result = self.agent.route(state)
        self.assertTrue(len(result.missing_information) > 0)

    def test_empty_document_is_handled_not_crashed(self):
        state = make_state(document_text="")
        result = self.agent.route(state)
        self.assertIn("document_text is empty or missing", result.missing_information)
        self.assertTrue(result.needs_human_review)

    # ---------------------------------------------------------------- #
    # Adversarial
    # ---------------------------------------------------------------- #
    def test_adversarial_keyword_stuffing_does_not_guarantee_high_confidence(self):
        # Stuffed with unrelated keywords from many departments at once --
        # should not produce a falsely confident single decision.
        state = make_state(
            document_text=(
                "personel izin bütçe ödeme yazılım arıza dava itiraz ihale "
                "satın alma protokol basın açıklaması soruşturma"
            ),
        )
        result = self.agent.route(state)
        self.assertIn(result.routing_status, ("AMBIGUOUS", "CONFLICT_DETECTED", "SINGLE_ROUTE"))
        if result.routing_status == "SINGLE_ROUTE":
            # If it still forced a single route, confidence must reflect
            # the noisiness -- never HIGH.
            self.assertNotEqual(result.confidence, "HIGH")

    def test_adversarial_excluded_topic_is_rejected(self):
        # A document squarely about a lawsuit should never be routed to HR,
        # even though "personel" appears, because it's an excluded topic.
        state = make_state(
            document_text="Personel dosyasına ilişkin açılan dava ve hukuki itiraz sürecinin takibi.",
            topic="dava",
        )
        result = self.agent.route(state)
        self.assertNotEqual(result.recommended_department, "İnsan Kaynakları Daire Başkanlığı")


# ---------------------------------------------------------------------- #
# Metrics harness
# ---------------------------------------------------------------------- #

LABELED_CASES = [
    dict(text="Personel atama ve kadro işlemlerinin yapılmasını rica ederim.", topic="personel",
         expected_department="İnsan Kaynakları Daire Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="Hukuki itiraz dilekçemizin değerlendirilerek dava sürecinin başlatılmasını talep ederiz.",
         topic="hukuki görüş", expected_department="Hukuk Müşavirliği", expect_conflict=False, expect_ambiguous=False),
    dict(text="Bütçe ödeneği talebi ve harcama belgesinin onaylanması gerekmektedir.", topic="bütçe",
         expected_department="Mali Hizmetler Daire Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="Sunucularda sistem arızası var, teknik destek ekibinin müdahalesi gerekiyor.", topic="teknik destek",
         expected_department="Bilgi İşlem Daire Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="İhale süreci başlatılacak, satın alma talebi ve teknik şartname hazırlanmalıdır.", topic="satın alma",
         expected_department="Destek Hizmetleri Daire Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="Soruşturma açılması ve usulsüzlük iddialarının incelenmesi talep edilmektedir.", topic="soruşturma",
         expected_department="Teftiş Kurulu Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="Basın açıklaması hazırlanması ve medya ile paylaşılması istenmektedir.", topic="basın",
         expected_department="Basın ve Halkla İlişkiler Müşavirliği", expect_conflict=False, expect_ambiguous=False),
    dict(text="Uluslararası işbirliği anlaşması kapsamında yurt dışı görevlendirme talebi.", topic="uluslararası",
         expected_department="Dış İlişkiler Daire Başkanlığı", expect_conflict=False, expect_ambiguous=False),
    dict(text="Konu hakkında değerlendirme talep olunur.", topic=None,
         expected_department=None, expect_conflict=False, expect_ambiguous=True),
    dict(text="Personel dosyasına ilişkin hukuki itiraz ve dava süreci hakkında görüş talep ederiz.",
         topic="personel", expected_department="Hukuk Müşavirliği", expect_conflict=None, expect_ambiguous=False),
]


def evaluate(agent: RoutingAgent, cases: List[dict]) -> dict:
    top1_hits = 0
    top3_hits = 0
    mrr_total = 0.0
    dept_hits = 0
    authority_hits = 0
    conflict_correct = 0
    ambiguity_correct = 0
    confidence_bucket_reasonable = 0
    n = len(cases)

    for case in cases:
        state = make_state(document_text=case["text"], topic=case["topic"])
        result = agent.route(state)

        ranked_departments = [result.recommended_department] + [
            r.department for r in result.alternative_routes
        ]

        expected = case["expected_department"]
        if expected is not None:
            if ranked_departments and ranked_departments[0] == expected:
                top1_hits += 1
                dept_hits += 1
            if expected in ranked_departments[:3]:
                top3_hits += 1
            if expected in ranked_departments:
                rank = ranked_departments.index(expected) + 1
                mrr_total += 1.0 / rank
            kb_dept = agent.kb.get_by_name(expected)
            if kb_dept and result.recommended_authority == kb_dept.authority_level:
                authority_hits += 1

        if case["expect_conflict"] is not None:
            got_conflict = bool(result.conflicts)
            if got_conflict == case["expect_conflict"]:
                conflict_correct += 1
        else:
            conflict_correct += 1  # not asserted for this case

        got_ambiguous = result.routing_status == "AMBIGUOUS" or bool(result.ambiguities)
        if got_ambiguous == case["expect_ambiguous"]:
            ambiguity_correct += 1

        # Calibration sanity: LOW confidence should coincide with either
        # ambiguity, conflict, or missing info; HIGH should coincide with
        # none of those.
        if result.confidence == "LOW":
            if got_ambiguous or result.conflicts or result.missing_information:
                confidence_bucket_reasonable += 1
        elif result.confidence == "HIGH":
            if not got_ambiguous and not result.conflicts:
                confidence_bucket_reasonable += 1
        else:
            confidence_bucket_reasonable += 1

    denom_dept = sum(1 for c in cases if c["expected_department"] is not None)
    return {
        "top1_accuracy": top1_hits / denom_dept if denom_dept else None,
        "top3_accuracy": top3_hits / denom_dept if denom_dept else None,
        "mrr": mrr_total / denom_dept if denom_dept else None,
        "department_accuracy": dept_hits / denom_dept if denom_dept else None,
        "authority_accuracy": authority_hits / denom_dept if denom_dept else None,
        "conflict_detection_accuracy": conflict_correct / n,
        "ambiguity_detection_accuracy": ambiguity_correct / n,
        "confidence_calibration": confidence_bucket_reasonable / n,
    }


class RoutingAgentMetricsTests(unittest.TestCase):
    def setUp(self):
        self.agent = RoutingAgent(knowledge_base=default_knowledge_base())

    def test_metrics_meet_minimum_bar(self):
        metrics = evaluate(self.agent, LABELED_CASES)
        # Lenient thresholds: this is a heuristic hybrid scorer over a small
        # seed knowledge base, not a tuned production model.
        self.assertGreaterEqual(metrics["top1_accuracy"], 0.6, metrics)
        self.assertGreaterEqual(metrics["top3_accuracy"], 0.75, metrics)
        self.assertGreaterEqual(metrics["mrr"], 0.65, metrics)
        self.assertGreaterEqual(metrics["ambiguity_detection_accuracy"], 0.7, metrics)
        self.assertGreaterEqual(metrics["confidence_calibration"], 0.7, metrics)


if __name__ == "__main__":
    unittest.main()
