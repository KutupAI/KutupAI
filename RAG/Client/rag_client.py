"""
هاد الملف الواجهة الوحيدة للتواصل مع الـ RAG الداخلي
"""

from RAG.client.retrieval_request import RetrievalRequest
from RAG.client.retrieval_response import RetrievalResponse
from RAG.retriever.context_formatter import extract_sources, format_context
from RAG.retriever.retriever import retrieve


def get_legal_context(request: RetrievalRequest) -> RetrievalResponse:
    """
    get_legal_context هي الدالة يلي بيتخزن فيها السؤال والجواب 
    نقطة الدخول الوحيدة لطبقة RAG بالكامل من منظور الـ Agents.

    Args:
        request: RetrievalRequest يحتوي على الـ query و top_k الاختياري.

    Returns:
        RetrievalResponse يحتوي على context جاهز للاستخدام، ومصادره،
        وعدد النتائج الفعلية.
    """
    results = retrieve(query=request.query, top_k=request.top_k)

    context = format_context(results)
    sources = extract_sources(results)

    return RetrievalResponse(
        context=context,
        sources=sources,
        result_count=len(results),
    )
