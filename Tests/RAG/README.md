# RAG Testleri

Bu klasör, RAG katmanının teslim öncesi testlerini içerir. Testler iki ayrı
akışa ayrılmıştır; retrieval skoru ile LLM cevabı birbirine karıştırılmaz.

## 1. Hızlı kod/regresyon testleri

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest Tests/RAG -q
python RAG/scripts/smoke_check.py --fast
```

## 2. Tam retrieval testi (LLM yok)

Held-out Türkçe hukuk sorularında **Hit@1/3/5, MRR, ortalama/p95 gecikme, RAM
ve indeks boyutunu** ölçer. Varsayılan olarak Chroma ve üç adil profil çalışır:
precision (vector+rereanker), balanced (hybrid+reranker), recall
(hybrid+PRF+reranker).

```powershell
python Tests/RAG/run_retrieval_evaluation.py
python Tests/RAG/run_retrieval_evaluation.py --backend faiss --profiles precision --build-faiss
```

## 3. Ana manuel test: LLM ile çok turlu hukukî sohbet

Inference/Qwen yerel servisi açıkken kullanıcı sorusuna kaynaklı cevap verir;
retrieval planını, kaynakları ve süreleri gösterir. İlişkili takip sorusunda
önceki soru yeniden aranmaz, yalnız aynı hukukî konu için yeni kanıt getirilir.

```powershell
python Tests/RAG/run_llm_evaluation.py
```

Bu dosya, kullanıcıların sistemi deneyip serbestçe soru sorması için **ana
test giriş noktasıdır**. Cevabı, seçilen retrieval yolunu, Query Transform ile
düzeltilmiş sorguları, kaynakları ve süreleri gösterir. `clear` yalnız sohbet
bağlamını sıfırlar; indeks veya kalıcı cevap cache'i silinmez.

### Query Transform seçenekleri

Varsayılan ayar hızlı ve yerel yazım düzeltmesidir; örneğin `nede` → `nerede`.
Ek Qwen tabanlı sorgu varyantlarını denemek için önce ayrı dönüşüm sunucusunu
başlatın:

```powershell
.\Inference\llama_server\query_transform_launcher.bat
```

Ardından aynı terminalden aşağıdakilerden birini kullanın:

```powershell
# Yalnız bu çalıştırma için LLM dönüşümünü açar.
python Tests/RAG/run_llm_evaluation.py --query-transform-llm

# Kalıcı ayar için RAG/configuration/rag_config.yaml içindeki
# query_transform.use_llm değerini true yapın.
python Tests/RAG/run_llm_evaluation.py
```

LLM dönüşüm sunucusu kapalıysa komut bunu başlangıçta bildirir; ana RAG sistemi
çalışmaya devam eder ve kural tabanlı dönüşüme geri döner.

Her iki testin ayrıntılı JSON çıktısı `RAG/evaluation/experiments/` altına
yazılır. Bu klasör yereldir ve Git'e eklenmez.

## Ayar değiştirme

Tüm çalışma ayarları, açıklamalarıyla birlikte
[`RAG/configuration/rag_config.yaml`](../../RAG/configuration/rag_config.yaml)
içindedir. Ayar değiştirildikten sonra benchmark'ı yeniden çalıştırın ve eski
sonuç dosyasını karşılaştırma için saklayın.
