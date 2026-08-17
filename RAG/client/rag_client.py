"""
KutupAI - RAG Client Interface
------------------------------
Agent katmanının iç RAG ile iletişim kurduğu tek istemci arabirimi.
PRF, reranker ve genişletme stratejisi gibi retrieval seçeneklerini destekler.
"""

from RAG.client.retrieval_request import RetrievalRequest
from RAG.client.retrieval_response import RetrievalResponse
from RAG.retriever.context_formatter import extract_sources, format_context
from RAG.retriever.retriever import retrieve


def get_legal_context(request: RetrievalRequest) -> RetrievalResponse:
    """
    Agent katmanı açısından RAG'a tek giriş noktasıdır.

    Args:
        request: Sorgu ve tüm retrieval seçeneklerini içeren istek.

    Returns:
        Kullanıma hazır bağlam, kaynakları ve gerçek sonuç sayısını içeren yanıt.
    """
    
    # Request içindeki tüm retrieval seçenekleri çekirdek arama fonksiyonuna aktarılır.
    results = retrieve(
        query=request.query, 
        top_k=request.top_k,
        mode=request.mode,
        use_prf=request.use_prf,
        use_reranker=request.use_reranker,
        expansion_strategy=request.expansion_strategy,
        where={"source_type": request.source_type} if request.source_type else None,
    )

    # Sonuçlar LLM'e verilebilecek izlenebilir bağlam metnine dönüştürülür.
    context = format_context(results)
    
    # Kaynak metadata'sı istemci sözleşmesine uygun biçimde döndürülür.
    sources = extract_sources(results)

    return RetrievalResponse(
        context=context,
        sources=sources,
        result_count=len(results),
    )
