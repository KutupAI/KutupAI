# RAG Temel Testleri

Bu klasör RAG katmanının teslim öncesi temel kontrollerini içerir. LLM ile
nihai cevap üretimi bu katmanın sorumluluğu olmadığı için burada test edilmez.

| Dosya | Kontrol ettiği alan |
|---|---|
| `test_layer_contract.py` | Katmanlar arası state sözleşmesi ve `rag.results` çıktısı |
| `test_pipeline.py` | Kaynakların yüklenmesi, parçalama ve indeksleme akışı |
| `test_ingestion_regressions.py` | Yeni kaynak eklenince ingestion davranışının bozulmaması |
| `test_retrieval.py` | Vektör/hybrid retrieval ile kaynaklı parça dönmesi |
| `test_query_router.py` | Sorunun uygun retrieval yoluna yönlendirilmesi |

Yerel, çok turlu LLM deneme aracı `run_llm_evaluation.py`'dır; otomatik test
değildir ve teslimde zorunlu değildir.

Tüm temel testleri çalıştırma:

```powershell
python -m pytest Tests/RAG/test_layer_contract.py Tests/RAG/test_pipeline.py Tests/RAG/test_ingestion_regressions.py Tests/RAG/test_retrieval.py Tests/RAG/test_query_router.py -q
```
