"""Yeni Layers_contracts formatı için tek RAG katman testi."""

from RAG.client.contract_adapter import handle_layer_state
from RAG.retriever.query_router import QueryPlan


def test_layer_contract_returns_retrieval_only(monkeypatch) -> None:
    """RAG, üst katman state'ini korur ve LLM olmadan yalnız kanıt paketini ekler."""
    def fake_retrieve(*args, **kwargs):
        assert kwargs["use_query_transform_llm"] is False
        return [{
            "id": "chunk-1",
            "text": "MADDE 1 - Deneme hükmü",
            "score": 0.9,
            "metadata": {"chunk_id": "chunk-1", "law_number": "1234", "law_name": "Deneme Kanunu", "article_no": "1", "page_start": 2, "page_end": 2},
        }]

    monkeypatch.setattr("RAG.client.contract_adapter.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "RAG.client.contract_adapter.choose_query_plan",
        lambda query: QueryPlan("semantic_fast", "vector", False, True, False, "test"),
    )
    state = {
        "request": {"success": True, "question": "bu ne sözleşmesi", "document": {"document_id": "DOC-1"}},
        "ocr": {"success": True, "ocr_data": {"full_text": "Elektrik aboneliği ve tüketim şartları"}},
        "classification": {"success": True, "document_type": "Elektrik sözleşmesi"},
        "extraction": {"success": True},
        "validation": {"success": True},
        "rag": {}, "summary": {}, "routing": {}, "writing": {},
    }
    result = handle_layer_state(state)
    assert result["request"] == state["request"]
    assert result["summary"] == {}
    assert result["rag"]["success"] is True
    assert result["rag"]["rag_data"]["operation"] == "retrieve"
    assert result["rag"]["rag_data"]["query"].startswith("bu ne sözleşmesi")
    source = result["rag"]["rag_data"]["results"][0]
    assert source == {
        "chunk_id": "chunk-1", "law_number": "1234", "law_name": "Deneme Kanunu",
        "article_no": "1", "page_start": 2, "page_end": 2,
        "text": "MADDE 1 - Deneme hükmü", "score": 0.9,
    }
    assert "answer" not in result["rag"]["rag_data"]
