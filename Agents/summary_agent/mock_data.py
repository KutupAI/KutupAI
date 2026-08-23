"""Sample inputs for manual/unit checks (not used in production)."""

MOCK_QUESTION = "Yabancı bir kişi Türkiye'de ikamet izni başvurusunu nereden yapar?"

MOCK_RAG_RESULT = {
    "success": True,
    "data": {
        "operation": "retrieve",
        "query": MOCK_QUESTION,
        "document_id": "doc-6458-001",
        "file_name": "ikamet_izni_basvurusu.pdf",
        "results": [
            {
                "chunk_id": "chunk-001",
                "text": (
                    "Türkiye'de ikamet izni başvurusu, yabancının bulunduğu ilin valiliğine "
                    "elektronik ortamda ve şahsen yapılır. Başvuru, geçerli bir pasaport veya "
                    "pasaport yerine geçen belge ile birlikte yapılmalıdır."
                ),
                "law_number": "6458",
                "article_no": "31",
                "article_type": "madde",
                "page_start": 12,
                "page_end": 13,
                "score": 0.91,
            },
            {
                "chunk_id": "chunk-002",
                "text": (
                    "İkamet izni başvurusu, yabancının Türkiye'ye girişinden itibaren doksan gün "
                    "içinde veya vizesinin ya da vize muafiyetinin süresi dolmadan yapılmalıdır. "
                    "Bu sürenin aşılması durumunda başvuru reddedilir."
                ),
                "law_number": "6458",
                "article_no": "31",
                "article_type": "madde",
                "page_start": 13,
                "page_end": 13,
                "score": 0.87,
            },
        ],
    },
    "error": None,
}

# Full GraphState-shaped fixture for SummaryAgent.run (Orchestration contract).
MOCK_STATE = {
    "request": {
        "success": True,
        "question": MOCK_QUESTION,
        "document": {
            "document_id": "doc-6458-001",
            "file_name": "ikamet_izni_basvurusu.pdf",
            "file_type": "pdf",
        },
    },
    "question": MOCK_QUESTION,
    "ocr": {},
    "classification": {},
    "extraction": {},
    "validation": {},
    # Orchestration stores RAG under rag_result; also set rag for pipeline-slot tests.
    "rag": {
        "success": True,
        "rag_data": {
            "operation": "retrieve",
            "query": MOCK_QUESTION,
            "results": MOCK_RAG_RESULT["data"]["results"],
        },
    },
    "rag_result": {
        "success": True,
        "data": {
            "operation": "retrieve",
            "query": MOCK_QUESTION,
            "results": MOCK_RAG_RESULT["data"]["results"],
        },
    },
    "summary": {},
    "routing": {},
    "writing": {},
}
