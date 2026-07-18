"""
rag_client.py
---------------
الواجهة الوحيدة التي يجب أن تستدعيها Agents Layer (rag_agent, writer_agent)
للتعامل مع طبقة RAG بالكامل. أي Agent لا يجب أن يستورد retriever.py أو
context_formatter.py أو vector_store مباشرة - فقط هذا الملف.

ملاحظة مهمة:
هذا الملف لا يستدعي Gemma / llama_client أبدًا. مسؤوليته الاسترجاع
والتنسيق فقط. استدعاء الـ LLM لتوليد الرد النهائي هو مسؤولية الـ Agent
نفسه (rag_agent / writer_agent) بعد استلام الـ RetrievalResponse من هنا.
"""

from RAG.client.retrieval_request import RetrievalRequest
from RAG.client.retrieval_response import RetrievalResponse
from RAG.retriever.context_formatter import extract_sources, format_context
from RAG.retriever.retriever import retrieve


def get_legal_context(request: RetrievalRequest) -> RetrievalResponse:
    """
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
