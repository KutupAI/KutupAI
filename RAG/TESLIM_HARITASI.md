# RAG Teslim Haritası

Bu dosya, RAG ekibinin GitHub tesliminde hangi dosyaların paylaşılacağını ve
hangi dosyaların yerel kalacağını açıklar.

## 1. GitHub'a eklenecek kaynak kod

| Alan | Klasör / dosya | Amaç |
|---|---|---|
| Ayarlar | `configuration/`, `requirements.txt` | RAG yapılandırması ve bağımlılıklar |
| Veri alma | `ingestion/`, `indexing/`, `metadata/` | PDF/TXT yükleme, chunking, metadata |
| Arama | `embeddings/`, `retriever/`, `vector_store/` | BGE embedding, BM25, hybrid, reranker, router, Chroma/FAISS/TurboVec adaptörleri |
| Agent | `agent/` | Context Builder, citation validator, semantic cache, Qwen agent |
| Graph | `graph/` | Hukuk maddesi ilişkileri ve Graph-RAG |
| Çalıştırma | `scripts/` | İnteraktif retrieval/agent, corpus audit, smoke check |
| Ölçüm | `evaluation/*.py` | Benchmark ve ablation scriptleri |
| Test | `../Tests/RAG/` | Pytest testleri |
| Dokümantasyon | `README.md`, `documents/*/README.md`, bu dosya | Kurulum ve çalışma açıklaması |

## 2. GitHub'a eklenecek küçük veri setleri

- `evaluation/datasets/heldout_legal.json`
- `evaluation/datasets/heldout_legal_ground_truth.json`
- `evaluation/datasets/evidence_qa_200.json`
- `evaluation/datasets/citation_lookup_legal.json`
- `evaluation/datasets/generation_legal.json`
- `evaluation/datasets/legal_safety_unanswerable.json`
- `evaluation/datasets/README.md`

Bu veri setleri test/benchmark içindir; kişisel veri veya kaynak PDF içermez.

## 3. Yerelde kalacak, GitHub'a eklenmeyecek dosyalar

| Yol | Neden |
|---|---|
| `documents/amendments/*.pdf` | Ham değişiklik kaynakları; küratörlü corpus dışında tutulur. |
| `documents/.chroma_db/`, `documents/.faiss/`, `documents/.turbovec/` | Yeniden üretilebilen indeksler. |
| `documents/manifest.json` | Yerel incremental-ingestion durumu; yeniden üretilir. |
| `documents/.semantic_cache.json` | Yerel çalışma verisi; eski cevap taşımamalıdır. |
| `documents/quarantine/` | Duplicate/kalite inceleme alanı. |
| `evaluation/experiments/`, `evaluation/models/`, `evaluation/*_render*/` | Büyük benchmark çıktıları, indirilen model ve render ara ürünleri. |

## 4. Teslim öncesi komutlar

```powershell
python -m pytest Tests/RAG -q
python RAG/scripts/smoke_check.py
.\RAG\scripts\run_legal_agent.ps1
```

Küratörlü `documents/laws/*.pdf`, `documents/regulations/*.pdf` ve
`documents/indexed_chunks.json` teslim corpusuna dahildir; `.gitattributes`
sayesinde Git LFS ile gönderilir. Bunlar yerel indekslerden farklı olarak
paylaşılabilir girdi ve incelenebilir chunk kaydıdır.

## 5. Sahiplik sınırı

Bu RAG teslimi yalnızca `RAG/` ve `Tests/RAG/` kapsamındadır. `Inference/`,
`Application/`, `Agents/`, `Orchestration/` ve `Presentation/` içindeki
değişiklikler ilgili ekiplerin sorumluluğundadır; RAG commit'ine dahil edilmez.
