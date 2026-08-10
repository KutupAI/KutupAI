# RAG Layer

## هيكل مبسّط

```
RAG/
├── documents/          ← ضع بياناتك هنا
├── ingestion/          ← رفع / تحميل / تقطيع / metadata / Chroma+BM25
├── retriever/          ← Hybrid + PRF + CrossEncoder + Query Expansion
├── client/             ← واجهة الـ Agents فقط
├── metadata/           ← schema + *.meta.json
├── evaluation/         ← Hit@k / MRR / dataset / QE experiment
├── embeddings/         ← BGE-M3
├── vector_store/       ← Chroma خلف واجهة ثابتة
├── chroma/             ← إعدادات Chroma
├── configuration/      ← rag_config.yaml
└── scripts/smoke_check.py
```

## أين تضع الداتا؟

| النوع | المجلد |
|---|---|
| قوانين | `RAG/documents/laws/` |
| لوائح | `RAG/documents/regulations/` |
| وثائق داخلية | `RAG/documents/internal_docs/` |
| رفع حر | `RAG/documents/uploads/` |

## شكل الداتا

**ملفات مدعومة:** `.txt` · `.md` · `.pdf`

**مثال نص:**
```text
MADDE 17- Belirsiz süreli iş sözleşmelerinin feshinden önce...
```

**Metadata اختياري** بجانب الملف بنفس الاسم:
`is_kanunu_4857.txt` + `is_kanunu_4857.meta.json`
```json
{
  "law_name": "4857 Sayılı İş Kanunu",
  "law_number": "4857",
  "effective_date": "2003-06-10",
  "source_type": "laws",
  "language": "tr",
  "tags": ["fesih", "ihbar"]
}
```

## تشغيل سريع

```bash
pip install -r RAG/requirements.txt

# بناء الفهرس من كل المجلدات
python -m RAG.ingestion.pipeline --reset

# ملف واحد
python -m RAG.ingestion.pipeline --file C:\path\doc.pdf --bucket laws
```

## اختبار بالمراحل
اختبار الطبقة بالمراحل
1) تثبيت

pip install -r RAG/requirements.txt
```bash
# 1) اختبار خفيف (بدون استرجاع ثقيل)
python Tests/RAG/test_pipeline.py

# 2) فحص كامل للطبقة
python RAG/scripts/smoke_check.py
python RAG/scripts/smoke_check.py --fast

# 3) Benchmark: Hit@1 Hit@2 Hit@3 MRR
python -m RAG.evaluation.benchmark

# 4) تجربة Query Expansion واختيار الأفضل
python -m RAG.evaluation.query_expansion_experiment

# 5) توليد أسئلة تقييم من الفهرس
python -m RAG.evaluation.dataset_generator --no-llm
```

## استخدام من الكود

```python
from RAG.client import RetrievalRequest
from RAG.client.rag_client import get_legal_context

resp = get_legal_context(RetrievalRequest(query="ihbar süreleri", top_k=5))
print(resp.context)
print(resp.sources)
```
BGE-M3	يحول النص إلى Embedding (Vector).
Chroma	يخزن الـ Embeddings ويبحث عنها دلاليًا.
BM25	يبحث بالكلمات المفتاحية.
Metadata	معلومات إضافية لتصفية وترتيب النتائج.
Hybrid Search	يجمع نتائج Chroma وBM25 للحصول على استرجاع أفضل.
Query Expansion (QE)	يوسّع السؤال بإضافة كلمات أو مرادفات لتحسين البحث.
PRF	يعيد صياغة البحث اعتمادًا على النتائج الأولى لتحسين الجولة الثانية.
Cross Encoder	يعيد ترتيب الوثائق المسترجعة ويختار الأكثر صلة.
Schema	يحدد شكل وهيكل البيانات المخزنة.
Dataset	مجموعة أسئلة وإجابات مرجعية لاختبار النظام.
Hit@k	يقيس هل الوثيقة الصحيحة ظهرت ضمن أول k نتائج.
MRR	يقيس مدى ارتفاع ترتيب الوثيقة الصحيحة في النتائج.
QE Experiment	تجربة تقارن أداء النظام قبل وبعد استخدام Query Expansion.
Chroma خلف واجهة ثابتة	تصميم برمجي يعزل بقية النظام عن تفاصيل قاعدة البيانات المتجهية.
smoke_check.py	اختبار سريع يتأكد أن جميع مكونات نظام RAG تعمل قبل التشغيل.