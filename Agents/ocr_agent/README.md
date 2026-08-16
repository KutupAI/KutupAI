# OCR Agent — دليل الطبقة الكامل

وكيل OCR يحوّل **صورة/PDF وثيقة** إلى **نص** للـ Orchestration.  
لا يصنّف، لا يستخرج كيانات أعمال، لا يستدعي RAG/Inference، ولا يكتب إلى Storage.

---

## وين بحط الملف؟

الـ OCR Agent **ما عنده مجلد رفع خاص**.  
هو يقرأ الملف من **مسار موجود على القرص** يصله عبر `graph_state`.

### 1) أين تخزّن الملف عملياً؟
عادةً Application يحفظ الرفع في:

```text
Storage/files/uploads/
```

مثال مسار كامل:

```text
C:\Users\...\SmartGovernmentAI\Storage\files\uploads\basvuru_001.pdf
```

الصيغ المدعومة فقط: `.pdf` · `.jpg` · `.jpeg` · `.png`

### 2) كيف يعرف الـ Agent مكان الملف؟
Orchestration يمرّر أحد هذه المفاتيح في الـ state:

| المفتاح | مثال |
|---|---|
| `document_path` | `"Storage/files/uploads/basvuru_001.pdf"` |
| `file_path` | نفس الفكرة |
| `input_path` | نفس الفكرة |
| `document.path` أو `document.file_path` | داخل كائن `document` |

مثال استدعاء:

```python
from Agents.ocr_agent import OCRAgent

state = {
    "document_id": "doc-001",
    "document_path": r"Storage\files\uploads\basvuru_001.pdf",
}

result_state = OCRAgent().run(state)
```

> مهم: ضع **مسار ملف حقيقي موجود**. الـ Agent لا يستقبل bytes من الواجهة مباشرة؛ Application يحفظ الملف أولاً ثم يمرّر المسار.

---

## وين بتعطيني النتيجة النهائية؟

النتيجة **ما بتنحفظ بملف تلقائياً**.  
ترجع داخل نفس الـ `state` اللي دخّلته لـ `run()`:

| المفتاح | المعنى |
|---|---|
| `ocr_status` | `"completed"` أو `"failed"` |
| `ocr_result` | النتيجة الكاملة (dict) — النص، الصفحات، الجداول، الأخطاء... |
| `document_text` | النص الكامل الجاهز للوكلاء التاليين (عند النجاح فقط) |
| `errors` | رسائل فشل (إن وجدت) |

### أهم جزء تحتاجه غالباً

```python
text = result_state["document_text"]          # النص النهائي
status = result_state["ocr_status"]           # completed / failed
full = result_state["ocr_result"]["full_text"]  # نفس النص داخل التقرير المفصّل
```

### شكل `ocr_result` (مختصر)

```json
{
  "success": true,
  "document_id": "doc-001",
  "file_name": "basvuru_001.pdf",
  "summary": {
    "has_signature": true,
    "has_handwritten_signature": true,
    "signature_names": ["Mehmet Kaya"],
    "primary_date": "10.08.2026",
    "dates": ["10.08.2026", "04.08.2026"],
    "has_articles": true,
    "article_count": 2,
    "line_count": 40
  },
  "has_signature": true,
  "has_handwritten_signature": true,
  "dates": ["10.08.2026"],
  "articles": [
    {"number": "1", "lines": ["..."], "text": "..."}
  ],
  "lines": ["سطر 1", "سطر 2"],
  "full_text": "...",
  "pages": [{"page_index": 0, "lines": ["..."], "text": "..."}]
}
```

ملاحظات سريعة:
- `document_id`: معرّف الوثيقة (من الـ state، أو اسم الملف بدون امتداد إذا لم يُمرَّر).
- `file_name`: اسم الملف الحقيقي مع الامتداد — منفصل عن `document_id`.
- للقراءة استخدم `lines` / `articles[].lines` (مصفوفات) بدل الاعتماد على `\n` داخل JSON.
- `has_handwritten_signature: true` عند وجود كتلة توقيع يدوي (مثل `İmza:` + اسم).

الوكلاء التاليين (Classification / Extraction / RAG...) يقرأون عادةً `document_text`.

---

## التدفق الكامل للطبقة

```text
[ملف على القرص]
        │
        ▼
 OCRAgent.run(state)          ← Agents/ocr_agent/agent.py
        │
        ▼
 OCRClient.process(...)       ← client.py  (واجهة ثابتة)
        │
        ▼
 OCRProcessor                 ← processing/processor.py
   1) validate_document       ← document.py
   2) تحميل الصفحات
        - PDF  → pdf_renderer.py (PyMuPDF)
        - صورة → OpenCV
   3) preprocessing           ← preprocessing/image_preprocessor.py
   4) Paddle / PP-StructureV3 ← engines/paddle_engine.py
   5) تحليل النتائج
        - نص      → core/ocr_parser.py
        - تخطيط   → core/layout.py
        - جداول   → core/tables.py
        - تصحيح تركي → core/correction.py
        │
        ▼
 UnifiedOCRResult             ← models.py
        │
        ▼
 state["ocr_result"]
 state["ocr_status"]
 state["document_text"]       ← النتيجة النهائية للـ Orchestration
```

---

## هيكل المجلد

```text
Agents/ocr_agent/
├── agent.py                 # BaseAgent + @register — نقطة Orchestration
├── client.py                # OCRClient / OCRRequest
├── tools.py                 # run_ocr() مساعدة
├── config.py                # OCRConfig من env
├── config.example.env       # أمثلة متغيرات البيئة
├── document.py              # التحقق من المسار/الصيغة/الحجم
├── models.py                # UnifiedOCRResult وبنية الصفحات
├── exceptions.py
├── prompts.py               # فارغ عمداً (لا LLM داخل OCR)
├── engines/
│   └── paddle_engine.py     # تحميل PP-StructureV3 مرة واحدة وإعادة استخدامه
├── preprocessing/
│   └── image_preprocessor.py
├── processing/
│   ├── processor.py         # خط الأنابيب الرئيسي
│   └── pdf_renderer.py
├── core/
│   ├── ocr_parser.py
│   ├── layout.py
│   ├── tables.py
│   └── correction.py
├── requirements.txt
└── README.md
```

---

## إعداد التشغيل

```bash
pip install -r Agents/ocr_agent/requirements.txt
```

متغيرات اختيارية: `config.example.env`  
(مثل `OCR_LANGUAGE=tr`, `OCR_DEVICE=cpu`, `OCR_PDF_DPI=300`)

أول تشغيل قد يحمّل نماذج PaddleOCR — هذا طبيعي.  
على Windows إذا فشل Paddle (oneDNN)، يتحول تلقائياً إلى **RapidOCR (ONNX)**.

### صور الهاتف الصعبة (بعيدة / مائلة / شاحبة)

الـ preprocessing يعالج تلقائياً:
- قص صفحة الوثيقة من الخلفية (`OCR_AUTO_CROP`)
- تصحيح منظور الميل/keystone + deskew حتى ~35°
- تعزيز التباين للألوان الشاحبة (`OCR_PALE_BOOST`)
- تكبير النص الصغير (`OCR_MIN_DIMENSION=1800`) ورفع DPI للـ PDF إلى 300

---

## مثال سريع من طرف لطرف

```python
from Agents.ocr_agent import OCRAgent

state = {
    "document_id": "doc-001",
    "document_path": r"Storage\files\uploads\ornek.pdf",
}

out = OCRAgent().run(state)

if out["ocr_status"] == "completed":
    print(out["document_text"])           # ← النتيجة النهائية هنا
else:
    print(out["ocr_result"])              # تفاصيل الفشل
    print(out.get("errors"))
```

أو بدون Agent مباشرة عبر الـ client:

```python
from Agents.ocr_agent import OCRClient

result = OCRClient().process_file(r"Storage\files\uploads\test_file.pdf")
print(result.full_text)                   # ← النص النهائي
print(result.to_dict())                   # ← التقرير الكامل
```

---


## الاختبارات (بدون تحميل نماذج)

```bash
python Tests/Agents/test_ocr_agent.py
# أو
pytest Tests/Agents/test_ocr_agent.py -q
```

الاختبارات تستخدم mock للمحرك؛ لا تحتاج GPU ولا تنزيل موديلات.

---

## حدود الطبقة (مهم)

| يفعل | لا يفعل |
|---|---|
| قراءة PDF/صورة من مسار | استقبال رفع HTTP مباشرة |
| OCR + جداول + تخطيط | تصنيف نوع الوثيقة |
| تصحيح تركي بسيط للنص | كتابة نتائج إلى Storage |
| إرجاع `ocr_result` في state | استدعاء RAG أو Inference |

للتشغيل الطبقة كاملة ويعطيني النتيجة 

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI

python -c "import json from pathlib import Path from Agents.ocr_agent import OCRClient
result = OCRClient().process_file(r'Storage\files\uploads\test_file.pdf')
out = Path(r'Storage\files\processed')
out.mkdir(parents=True, exist_ok=True)
data = result.to_dict()
(out / 'test_file.ocr.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
(out / 'test_file.ocr.txt').write_text('\n'.join(data['lines']), encoding='utf-8')

print(data['summary'])
print('TXT  ->', out / 'test_file.ocr.txt')
print('JSON ->', out / 'test_file.ocr.json')
"

أو عبر الـ Agent:

python -c "from Agents.ocr_agent import OCRAgent out = OCRAgent().run({ 'document_id': 'doc-001', 'document_path': r'Storage\files\uploads\test_file.pdf',})
print(out['ocr_status'])
print(out['ocr_result']['summary'])
"