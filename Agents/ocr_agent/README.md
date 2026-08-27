# OCR Agent

Converts **image / PDF / DOCX / PPTX / XLSX** into a structured OCR result for Orchestration.
Does **not** classify, extract business entities, call RAG, or write to Storage.

Engines are **on-demand**: RapidOCR / PaddleOCR-VL / PP-StructureV3 run only when
PaddleOCR is not enough (or a table is detected). Skipping them means the page did
**not need them** — not that those engines are broken.

---

## مخطط عمل الطبقة

```text
┌─────────────────────────────────────────────────────────────────┐
│ Orchestration (أو الاختبار المستقل)                              │
│   state["request"]["document"]  +  مسار الملف                     │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ OCRAgent.run(state)                                             │
│   resolve path → OCRClient → OCRProcessor                       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Per document                                                    │
│   validate → normalize (PDF / image / office)                   │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Per page                                                        │
│   quality_analyzer → preprocessing (light / full)               │
│                                                                 │
│   Cascade (stop early when result is Good or Usable):           │
│                                                                 │
│     1) PaddleOCR          → Good/Usable?  ACCEPT (done)         │
│     2) RapidOCR           → فقط إذا Paddle رجّع فارغ              │
│     3) PaddleOCR-VL       → فقط إذا النص ما زال فاضي/ناقص         │
│     4) PP-StructureV3     → فقط إذا كُشف جدول في الصفحة           │
│                                                                 │
│   OCR_MAX_ATTEMPTS يعيد preprocess أقوى قبل VL عند الحاجة         │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ result_builder                                                  │
│   state["ocr"]           ← العقد الموحّد للطبقات                 │
│   state["ocr_result"]    ← { Success, Data } لـ Orchestration    │
│   state["ocr_status"]    ← completed | partial | failed         │
│   state["document_text"] ← full_text عند النجاح                  │
└─────────────────────────────────────────────────────────────────┘
```

**ملاحظة:** صفحات PDF ذات طبقة نص أصلية قابلة للاستخدام تتخطى raster OCR.

PaddleOCR يحمّل فقط نماذج خفيفة: `PP-OCRv5_mobile_det` + `PP-OCRv5_mobile_rec`  
(لا `server_det` / `latin_rec` / `PP-LCNet` / `DocLayout_plus-L` على مسار النص العادي).

---

## التكامل مع Orchestration

الطبقة **مربوطة فعلياً** ومفعّلة في المشروع الكامل:

| المكان | التفاصيل |
|--------|----------|
| `Orchestration/config.yaml` | `agents.ocr.enabled: true` → `Agents.ocr_agent.OCRAgent` |
| `Orchestration/workflow/workflow_config.py` | مرحلة `Stage.OCR` افتراضياً مفعّلة |
| `Orchestration/process_service.py` | `run_workflow` يشغّل الـ Graph؛ OCR مرحلة أولى مثل باقي الـ Agents |
| `Orchestration/graph` | المراحل التالية تقرأ `ocr_result` كمدخل مطلوب |

عند تشغيل المشروع كاملاً:

```text
Application / API
    → POST /process
    → Orchestration.run_workflow
        → Stage.OCR → OCRAgent().run(state)
            → state["ocr"] + state["ocr_result"] + state["ocr_status"]
        → مراحل لاحقة إن كانت enabled
```

- اليوم: OCR هو الـ Agent الوحيد المفعّل؛ باقي المراحل `enabled: false` وتُتخطى بأمان.
- لا يوجد مسار OCR منفصل خارج الـ Graph.
- الاختبار المستقل لطبقة OCR يستخدم نفس شكل الـ state بدون تشغيل بقية الـ Graph.

عقد Orchestration السلكي من `OCRAgent`:

- `ocr_result` → `{ Success, Data: [document, ...] }`
- `ocr_status` → `completed` | `partial` | `failed`
- `document_text` → النص الكامل عند النجاح
- `ocr` → العقد الموحّد التفصيلي للطبقات

---

## أوامر التشغيل والاختبار

شغّل الأوامر من **جذر المشروع** (`SmartGovernmentAI/`):

### 1) مسار OCR الكامل (الملف الذي كنا نجرّب عليه)

```powershell
python Tests/Agents/test_ocr_agent_standalone.py --file "Storage/files/uploads/clear.jpg"
```

أمثلة أخرى:

```powershell
python Tests/Agents/test_ocr_agent_standalone.py --file "Storage/files/uploads/Elektrik sozlesmesi.pdf"
python Tests/Agents/test_ocr_agent_standalone.py --document-id DOC-001 --file-name "Elektrik sozlesmesi.pdf"
```

### 2) فحص PaddleOCR-VL فقط (منفذ 8111)

أولاً شغّل السيرفر المحلي:

```powershell
Inference\start_paddleocr_vl.bat
```

ثم:

```powershell
python Tests/Agents/test_ocr_agent_standalone.py --file "Storage/files/uploads/scan.pdf" --force-vision-fallback
```

### ماذا يطبع الاختبار؟

- `state` الموحّد دخلاً وخرجاً
- ملخص: `success` / `status` / `page_count` / مقطع من `full_text`
- يتحقق أن أقسام Orchestration الأخرى (`classification`, `extraction`, …) لم تُمس

---

## Unified I/O

يقرأ `state["request"]` فقط؛ يكتب `state["ocr"]` (+ مفاتيح الـ wire أعلاه). بقية الأقسام تمر كما هي.

```python
from Agents.ocr_agent import OCRAgent

state = {
    "request": {
        "success": True,
        "question": "bu ne sozlesmesi",
        "document": {
            "document_id": "DOC-001",
            "file_name": "Elektrik sozlesmesi.pdf",
            "file_type": "pdf",
        },
    },
    "ocr": {}, "classification": {}, "extraction": {}, "validation": {},
    "rag": {}, "summary": {}, "routing": {}, "writing": {},
}
result_state = OCRAgent().run(state)
print(result_state["ocr"])
```

### حل مسار الملف

1. مسار مباشر: `document_path` / `file_path` / `input_path` أو `document.path`
2. وإلا `Storage.file_locator.resolve_upload_path(...)` إن وُجد
3. وإلا `<OCR_STORAGE_UPLOADS_DIR>/<file_name>` (افتراضي `Storage/files/uploads/`)

**الصيغ:** صور (jpg/png/webp/tiff/bmp/gif)، PDF، نصوص مكتبية (docx/pptx/xlsx — الصور المضمّنة لا تُقرأ بـ OCR).

---

## Output (`state["ocr"]`)

```json
{
  "success": true,
  "status": "complete",
  "ocr_data": {
    "page_count": 1,
    "language": "tr",
    "pages": [
      {
        "page_number": 1,
        "text": "...",
        "vision": {
          "signature": {"detected": true, "handwritten": true},
          "stamp": {"detected": false}
        }
      }
    ],
    "full_text": "...",
    "vision": {
      "signature": {"detected": true, "handwritten": true},
      "stamp": {"detected": false}
    }
  }
}
```

`status`: `complete` | `partial` | `failed`

**Error codes:** `UNSUPPORTED_FILE_TYPE`, `FILE_CORRUPTED`, `PAGE_EXTRACTION_FAILED`, `OCR_FAILED`, `LOW_IMAGE_QUALITY`, `LOW_OCR_CONFIDENCE`, `VISION_FALLBACK_FAILED`, `SIGNATURE_DETECTION_FAILED`, `SEAL_DETECTION_FAILED`.

### متى يُستدعى كل محرك؟

| المحرك | يشتغل متى؟ | تخطّيه يعني |
|--------|------------|-------------|
| **PaddleOCR** | دائماً للنص (أساسي، GPU) | — |
| **RapidOCR** | فقط إذا Paddle رجّع **فارغ** | الصفحة صارت Usable/Good من Paddle |
| **PaddleOCR-VL** | فقط إذا بعد Paddle(+Rapid) النص ما زال فاضي/ناقص جداً | الصفحة واضحة وكافية |
| **PP-StructureV3** | فقط إذا `enable_tables` وكُشف شكل جدول | الصفحة مش جدول |

مثال `scan.pdf` الأخير: Paddle رجّع نص قوي → Rapid/VL/Structure = **0 ث** عمداً (المسار الصحيح)، مو عطل.

### متى يُستدعى PaddleOCR-VL؟

فقط إذا كان مفعّلاً وبعد أن يبقى النص فاضي أو ناقص بعد Paddle (وRapid إن لزم). الصفحات الواضحة لا تستدعيه.

---

## Engines & device

| الترتيب | المحرك | الدور | Good / Bad |
|---------|--------|--------|------------|
| 1 | PaddleOCR (`mobile_det` + `mobile_rec`) | نص عادي (أساسي) | نعم (+ Usable) |
| 2 | RapidOCR (ONNX) | فقط عند فراغ نتيجة Paddle | نعم |
| 3 | PaddleOCR-VL | آخر ملاذ للنص الناقص | نعم |
| 4 | PP-StructureV3 Table Pipeline | جداول فقط عند الكشف | نعم |

- `OCR_DEVICE=gpu` (افتراضي): GPU إن وُجد، وإلا CPU
- المحركات **lazy + singleton** داخل عملية Orchestration (warm-up عند الإقلاع)
- Structure لا يُحمَّل إلا عند الحاجة للجداول

```bash
pip install -r Agents/ocr_agent/requirements.txt
```

- **GPU (موصى به):** ثبّت `paddlepaddle-gpu` المطابق لـ CUDA
- **CPU:** `paddlepaddle>=3.1,<3.3` (يفضّل `3.2.2`)
- ادمج `config.example.env` في `.env` المشروع

---

## هيكل المجلد

```text
Agents/ocr_agent/
├── agent.py                 # BaseAgent — نقطة دخول Orchestration
├── client.py
├── config.py
├── device.py
├── document.py
├── exceptions.py
├── models.py
├── engines/paddle_engine.py
├── preprocessing/
├── pipeline/
├── interfaces/
├── processing/
├── core/
├── requirements.txt
├── config.example.env
└── README.md
```

---

## حدود الطبقة

| يفعل | لا يفعل |
|------|----------|
| OCR + تخطيط/جداول/إشارات توقيع | تصنيف الوثيقة / استخراج حقول عمل |
| معالجة مسبقة + إعادة محاولة | تشغيل VL على كل صفحة |
| VL اختياري عند الحاجة | استقبال رفع HTTP مباشرة |
| إرجاع `state["ocr"]` + wire keys | الكتابة إلى Storage |

انشاء بيئىة افتراضية وتشغيلها:
cd /d D:\AI\KutupAI
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r Agents\ocr_agent\requirements.txt