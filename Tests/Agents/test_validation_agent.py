"""
Tests for Agents/validation_agent, rewritten against the real project
contract (Layers_contracts/Agents-contract/Validate-contract.md).

Replaces the previous test suite, which was written against an invented
state shape (extracted_data / document_type / TCKN / VKN) that does not
exist in the real project.
"""
import json

from Agents.validation_agent.agent import ValidationAgent
from Agents.validation_agent import tools


def make_agent() -> ValidationAgent:
    return ValidationAgent()


def base_state(**overrides) -> dict:
    """
    Builds a shared-state dict matching the real contract shape, with the
    exact canonical example from Validate-contract.md as the default
    extraction/classification/ocr payload. Individual namespaces can be
    overridden per test.
    """
    state = {
        "request": {
            "success": True,
            "question": "bu ne sozlesmesi",
            "document": {
                "document_id": "DOC-001",
                "file_name": "Elektrik sozlesmesi.pdf",
                "file_type": "pdf",
            },
        },
        "ocr": {
            "success": True,
            "ocr_data": {
                "page_count": 1,
                "language": "tr",
                "pages": [],
                "full_text": "...",
                "vision": {
                    "signature": {"detected": True, "handwritten": True},
                    "stamp": {"detected": False},
                },
            },
        },
        "classification": {
            "success": True,
            "document_type": "Elektrik sozlesmesi",
            "classification_confidence": 0.95,
        },
        "extraction": {
            "success": True,
            "sender": None,
            "date": None,
            "address": None,
            "phone": None,
            "email": None,
        },
        "validation": {},
        "rag": {},
        "summary": {},
        "routing": {},
        "writing": {},
    }
    state.update(overrides)
    return state


def test_valid_pipeline_state_matches_contract_example():
    agent = make_agent()
    state = base_state()

    result_state = agent.run(state)

    import json
    print(json.dumps(result_state, indent=2, ensure_ascii=False))

    result = result_state["validation"]

    print("validation output:", result)

    assert result == {
        "success": True,
        "is_complete": False,
        "errors": [],
        "warnings": [],
    }


def test_extraction_success_with_full_data_marks_complete():
    state = base_state(extraction={
        "success": True,
        "sender": "Ahmet Yilmaz",
        "date": "01.03.2026",
        "address": "Istanbul",
        "phone": "5551234567",
        "email": "ahmet@example.com",
    })
    result = make_agent().run(state)["validation"]
    assert result["success"] is True
    assert result["is_complete"] is True
    assert result["errors"] == []


def test_extraction_failure_flagged_as_error():
    state = base_state(extraction={
        "success": False,
        "sender": None,
        "date": None,
        "address": None,
        "phone": None,
        "email": None,
    })
    result = make_agent().run(state)["validation"]
    assert "extraction_failed" in result["errors"]
    assert result["success"] is False
    assert result["is_complete"] is False


def test_partial_extraction_flags_warning():
    state = base_state(extraction={
        "success": True,
        "sender": "Ahmet Yilmaz",
        "date": None,
        "address": None,
        "phone": None,
        "email": None,
    })
    result = make_agent().run(state)["validation"]
    assert "partial_extraction_data" in result["warnings"]
    assert result["errors"] == []
    assert result["is_complete"] is False


def test_empty_extraction_no_warning_no_error():
    state = base_state()  # default extraction is all-null, success True
    result = make_agent().run(state)["validation"]
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["is_complete"] is False
    assert result["success"] is True


def test_invalid_date_format_when_present():
    state = base_state(extraction={
        "success": True,
        "sender": None,
        "date": "not-a-date",
        "address": None,
        "phone": None,
        "email": None,
    })
    result = make_agent().run(state)["validation"]
    assert "invalid_date_format" in result["errors"]


def test_date_not_checked_when_absent():
    state = base_state()
    result = make_agent().run(state)["validation"]
    assert "invalid_date_format" not in result["errors"]


def test_invalid_email_format_when_present():
    state = base_state(extraction={
        "success": True,
        "sender": None,
        "date": None,
        "address": None,
        "phone": None,
        "email": "not-an-email",
    })
    result = make_agent().run(state)["validation"]
    assert "invalid_email_format" in result["errors"]


def test_invalid_phone_format_when_present():
    state = base_state(extraction={
        "success": True,
        "sender": None,
        "date": None,
        "address": None,
        "phone": "123",
        "email": None,
    })
    result = make_agent().run(state)["validation"]
    assert "invalid_phone_format" in result["errors"]


def test_valid_phone_format_with_country_code():
    state = base_state(extraction={
        "success": True,
        "sender": None,
        "date": None,
        "address": None,
        "phone": "+90 555 123 45 67",
        "email": None,
    })
    result = make_agent().run(state)["validation"]
    assert "invalid_phone_format" not in result["errors"]


def test_low_classification_confidence_warns():
    state = base_state(classification={
        "success": True,
        "document_type": "Elektrik sozlesmesi",
        "classification_confidence": 0.2,
    })
    result = make_agent().run(state)["validation"]
    assert "low_classification_confidence" in result["warnings"]
    assert result["success"] is True


def test_classification_failure_warns():
    state = base_state(classification={
        "success": False,
        "document_type": None,
        "classification_confidence": None,
    })
    result = make_agent().run(state)["validation"]
    assert "classification_failed" in result["warnings"]
    assert result["success"] is True


def test_ocr_failure_warns():
    state = base_state(ocr={"success": False, "ocr_data": {}})
    result = make_agent().run(state)["validation"]
    assert "ocr_failed" in result["warnings"]
    assert result["success"] is True


def test_empty_ocr_text_warns():
    state = base_state(ocr={
        "success": True,
        "ocr_data": {
            "page_count": 1,
            "language": "tr",
            "pages": [],
            "full_text": "",
            "vision": {"signature": {"detected": False, "handwritten": False}, "stamp": {"detected": False}},
        },
    })
    result = make_agent().run(state)["validation"]
    assert "empty_ocr_text" in result["warnings"]


def test_shared_state_preserved():
    state = base_state()
    original_request = dict(state["request"])
    original_ocr = dict(state["ocr"])
    original_classification = dict(state["classification"])
    original_extraction = dict(state["extraction"])

    result_state = make_agent().run(state)

    assert result_state["request"] == original_request
    assert result_state["ocr"] == original_ocr
    assert result_state["classification"] == original_classification
    assert result_state["extraction"] == original_extraction
    assert result_state["rag"] == {}
    assert result_state["summary"] == {}
    assert result_state["routing"] == {}
    assert result_state["writing"] == {}
    assert "validation" in result_state


def test_validation_output_schema_exact_keys():
    state = base_state()
    result = make_agent().run(state)["validation"]
    assert set(result.keys()) == {"success", "is_complete", "errors", "warnings"}
    assert isinstance(result["success"], bool)
    assert isinstance(result["is_complete"], bool)
    assert isinstance(result["errors"], list)
    assert isinstance(result["warnings"], list)
    for forbidden_key in ("status", "confidence", "checked_rules", "missing_fields", "semantic_note", "invalid_fields"):
        assert forbidden_key not in result


def test_validation_routing_compatibility():
    state = base_state(extraction={
        "success": True,
        "sender": "Ahmet Yilmaz",
        "date": "01.03.2026",
        "address": "Istanbul",
        "phone": "5551234567",
        "email": "ahmet@example.com",
    })
    result_state = make_agent().run(state)

    validation = result_state["validation"]
    assert validation["success"] is True
    assert validation["is_complete"] is True

    assert result_state["classification"]["document_type"] == "Elektrik sozlesmesi"
    assert result_state["extraction"]["success"] is True
    assert result_state["ocr"]["success"] is True
    assert result_state["rag"] == {}
    assert result_state["routing"] == {}


def test_agent_registered_with_correct_name():
    assert ValidationAgent.name == "validation_agent"


def test_validate_date_format_unit():
    assert tools.validate_date_format("01.03.2026") is True
    assert tools.validate_date_format("2026-03-01") is True
    assert tools.validate_date_format("not-a-date") is False
    assert tools.validate_date_format(None) is False


def test_validate_email_format_unit():
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    assert tools.validate_email_format("a@b.com", pattern) is True
    assert tools.validate_email_format("not-an-email", pattern) is False
    assert tools.validate_email_format(None, pattern) is False


def test_validate_phone_format_unit():
    assert tools.validate_phone_format("5551234567", 10) is True
    assert tools.validate_phone_format("+905551234567", 10) is True
    assert tools.validate_phone_format("05551234567", 10) is True
    assert tools.validate_phone_format("123", 10) is False
    assert tools.validate_phone_format(None, 10) is False


def test_missing_extraction_key_does_not_crash():
    state = base_state()
    del state["extraction"]
    result = make_agent().run(state)["validation"]
    assert result["success"] is True
    assert "extraction_result_missing" in result["warnings"]


def test_completely_empty_state_does_not_crash():
    result = make_agent().run({})["validation"]
    assert result["success"] is True
    assert result["is_complete"] is False