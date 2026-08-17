# OCR Agent — دليل الطبقة الكامل

وكيل OCR يحوّل **صورة/PDF/DOCX/PPTX/XLSX** إلى **نتيجة OCR منظّمة** للـ Orchestration.
لا يصنّف، لا يستخرج كيانات أعمال، لا يستدعي RAG/Inference (عدا Vision Fallback عند الحاجة فقط)، ولا يكتب إلى Storage.

---

## أين أضع الملف؟

الـ OCR Agent **ما عنده مجلد رفع خاص**. يقرأ الملف من مسار موجود على القرص يصله عبر `graph_state`
(عادة `Storage/files/uploads/...`، حسب `README.md`/`architecture.md` للمشروع).

Orchestration يمرّر أحد هذه المفاتيح في الـ state:

| المفتاح | مثال |
|---|---|
| `document_path` | `"Storage/files/uploads/basvuru_001.pdf"` |
| `file_path` | نفس الفكرة |
| `input_path` | نفس الفكرة |
| `document.path` أو `document.file_path` | داخل كائن `document` |

```python
from Agents.ocr_agent import OCRAgent

state = {"document_id": "doc-001", "document_path": "Storage/files/uploads/basvuru_001.pdf"}
result_state = OCRAgent().run(state)
```

الصيغ المدعومة:
- صور: `.jpg .jpeg .png .webp .tiff .tif .bmp .gif`
- مستندات: `.pdf`
- مكتبية (نص فقط، بدون OCR للصور المضمّنة داخلها): `.docx .pptx .xlsx`

---

## عقد الإخراج (Output Contract)

النتيجة ترجع داخل `state["ocr_result"]` (و`state["ocr_status"]` = `complete|partial|failed`).
الشكل ثابت ولا يحتوي أي حقول دلالية (`classification`, `extracted_data`, `summary`, `answer`, ...) — تلك تخص طبقات لاحقة.

```json
{
  "success": true,
  "status": "complete",
  "data": {
    "document_id": "doc-001",
    "file_name": "basvuru_001.pdf",
    "file_type": "pdf",
    "page_count": 2,
    "language": {"detected": "tr", "confidence": 0.94},
    "pages": [
      {
        "page_number": 1,
        "text": "...",
        "blocks": [
          {"type": "title", "text": "T.C. İstanbul Valiliği", "bbox": [100,80,700,150], "confidence": 0.98, "uncertain": false}
        ],
        "tables": [{"bbox": [0,0,0,0], "confidence": 0.94, "rows": [["Sütun 1","Sütun 2"], ["Değer 1","Değer 2"]]}],
        "vision": {
          "signature": {"detected": true, "handwritten": true, "confidence": 0.94, "bbox": [0,0,0,0]},
          "stamp": {"detected": true, "confidence": 0.97, "bbox": [0,0,0,0], "text": null}
        },
        "quality": {"score": 0.88, "ocr_confidence": 0.95, "readable": true},
        "warnings": []
      }
    ],
    "full_text": "...",
    "processing": {
      "ocr_engine": "PP-OCRv5",
      "structure_engine": "PP-StructureV3",
      "engines_used": ["PaddleOCR + PP-StructureV3"],
      "fallback_used": false,
      "pages_reprocessed": [],
      "duration_ms": 812.3
    }
  }
}
```

On failure (bad path, unsupported type, corrupted file):

```json
{"success": false, "status": "failed", "error": {"code": "UNSUPPORTED_FILE_TYPE", "message": "..."}, "data": {"...": "empty shell"}}
```

`status`:
- `complete` — every page produced text.
- `partial` — some pages succeeded, some didn't (never discards the good pages).
- `failed` — document could not be meaningfully processed.

Downstream agents should read `state["document_text"]` (plain `full_text`, set only on success) or walk `state["ocr_result"]["data"]["pages"]` for structure.

### Error codes
`UNSUPPORTED_FILE_TYPE`, `FILE_CORRUPTED`, `PAGE_EXTRACTION_FAILED`, `OCR_FAILED`,
`LOW_IMAGE_QUALITY`, `LOW_OCR_CONFIDENCE`, `VISION_FALLBACK_FAILED`,
`SIGNATURE_DETECTION_FAILED`, `SEAL_DETECTION_FAILED`.

---

## التدفق التكيّفي (Adaptive pipeline)

```
input → validate → normalize (pdf/image/office) → per-page:
  quality_analyzer (blur/brightness/contrast)
     good → light/no preprocessing → OCR attempt 1
     poor → full adaptive preprocessing → OCR attempt 1
  confidence_analyzer decides:
     ACCEPT   → done
     RETRY    → stronger preprocessing → OCR attempt N (up to OCR_MAX_ATTEMPTS)
     FALLBACK → Qwen-VL vision fallback (only if enabled, only for that page)
     GIVE_UP  → best-effort result, flagged uncertain
→ result_builder → output contract above
```

For PDFs, the native-text-vs-OCR decision is **per page**, not per document —
a cover letter with real text plus a scanned attachment in the same PDF gets
both pages handled correctly.

### بالعربي — متى يُستدعى Qwen-VL بالضبط؟

**الصورة واضحة (جودة جيدة + ثقة OCR عالية):**
`quality_analyzer` يقيس الوضوح قبل أي OCR، فإذا كانت النتيجة فوق `OCR_QUALITY_THRESHOLD`
يتخطى المعالجة المسبقة الثقيلة. وبعد أول محاولة OCR، إذا كانت الثقة فوق
`OCR_LOW_CONFIDENCE_THRESHOLD` يقبل النتيجة فوراً (`ACCEPT`) — **لا يُستدعى Qwen-VL إطلاقاً**، لا داعي له.

**الصورة غير واضحة:**
1. تُطبَّق معالجة مسبقة كاملة (deskew, perspective correction, تباين, إزالة تشويش...) ثم محاولة OCR ثانية (`RETRY`)، إلى أن يصل عدد المحاولات لـ `OCR_MAX_ATTEMPTS` (افتراضياً 3).
2. إذا استُنفدت كل المحاولات وبقيت الثقة **تحت** `OCR_VISION_FALLBACK_THRESHOLD` (افتراضياً 0.45)، **و**كان `OCR_VISION_FALLBACK_ENABLED=true` في الإعدادات → عندها فقط يُستدعى Qwen-VL، لصفحة واحدة فقط، وليس للمستند كله.
3. إذا كان الفallback **معطّلاً** (وهذا هو الوضع الافتراضي حالياً) → يُقبل أفضل نتيجة OCR متاحة مع `uncertain: true` وتحذير في `warnings`، بدل الفشل الكامل.

باختصار: **واضحة → قبول مباشر بدون أي fallback. غير واضحة → إعادة محاولة أولاً، و Qwen-VL هو آخر خيار فقط بعد فشل كل المحاولات وبشرط تفعيله في الإعدادات.**

⚠️ **ملاحظة مهمة**: كما ذكرت بالتقرير، الفallback مُفعّل بنياً في الكود (الواجهة `interfaces/vision_fallback.py` جاهزة) لكنه **معطّل افتراضياً** (`OCR_VISION_FALLBACK_ENABLED=false`) لأنه يحتاج عميل Qwen-VL حقيقي متصل بخادم غير موجود عندك بعد في طبقة Inference. لتفعيله فعلياً تحتاج تشغّل خادم Qwen-VL وتربطه، وإلا سيبقى يُرجع خطأ `VISION_FALLBACK_FAILED` لو فعّلته بدون خادم حقيقي.

---

## الجهاز (Device) والأداء

`OCR_DEVICE=auto` (default) probes for a CUDA GPU at process start
(`device.py`) and falls back to CPU automatically; no CUDA id, GPU model or
path is hard-coded anywhere in the Agent. `OCR_PERFORMANCE_PROFILE`
(`development|production|high_performance`) is metadata only for now — it's
threaded through so behavior can be tuned via config later without code
changes.

The heavy PaddleOCR/PP-StructureV3 engine is a **process-wide singleton**
(`engines/paddle_engine.py::get_shared_engine`), cached by config. Re-creating
`OCRAgent()`/`OCRClient()` per Orchestration call (normal Supervisor
behavior) does **not** reload model weights.

---

## Vision Fallback (Qwen-VL)

Disabled by default (`OCR_VISION_FALLBACK_ENABLED=false`). When enabled, it's
called only for a page that is still low-confidence after all OCR retries,
capped at `OCR_VISION_FALLBACK_MAX_PAGES` pages per document. It never runs
on every page and it's lazy-loaded (no client/model handle is created until
first use).

`Agents/ocr_agent/interfaces/vision_fallback.py` defines
`VisionFallbackInterface` (swap providers without touching the pipeline) and
a `QwenVLVisionFallback` adapter that expects a
`Inference.client.llama_client.VisionInferenceClient` with a
`generate_vision(image_base64, instructions, timeout_s)` method. **That
client does not exist yet in this repository's Inference layer** (which
currently serves Gemma 3 via llama-server, not a vision-language model) —
see the final report for what's required to wire it up.

---

## هيكل المجلد

```text
Agents/ocr_agent/
├── agent.py                   # BaseAgent + @register — Orchestration entry point
├── client.py                  # OCRClient / OCRRequest (cached processor)
├── config.py                  # OCRConfig (env-driven, incl. device/quality/retry/fallback)
├── device.py                  # GPU auto-detect / device resolution
├── document.py                # path validation, supported extensions
├── exceptions.py              # OCRAgentError hierarchy with stable `code`s
├── models.py                  # BoundingBox / OCRTextItem / LayoutElement / TableResult / ...
├── engines/
│   └── paddle_engine.py       # PP-StructureV3 → PaddleOCR → RapidOCR, cached singleton
├── preprocessing/
│   └── image_preprocessor.py  # adaptive preprocessing (crop/deskew/contrast/...)
├── pipeline/
│   ├── quality_analyzer.py    # pre-OCR image quality scoring
│   └── confidence_analyzer.py # accept / retry / fallback decision
├── interfaces/
│   ├── vision_fallback.py     # VisionFallbackInterface + Qwen-VL adapter
│   └── signature_detector.py  # SignatureSealDetectorInterface + heuristic default
├── processing/
│   ├── processor.py           # orchestrates the whole pipeline
│   ├── pdf_renderer.py        # PyMuPDF: native text + rasterization
│   ├── office_renderer.py     # DOCX/PPTX/XLSX text extraction
│   └── result_builder.py      # builds the stable output contract
├── core/
│   └── ocr_parser.py, layout.py, tables.py, correction.py, insights.py
├── requirements.txt
└── README.md
```

---

## إعداد التشغيل

```bash
pip install -r Agents/ocr_agent/requirements.txt
cp Agents/ocr_agent/config.example.env .env   # or merge into project .env
```

## الاختبارات (بدون تحميل نماذج)

```bash
python Tests/Agents/test_ocr_agent.py
# or
pytest Tests/Agents/test_ocr_agent.py -q
```

The engine is mocked (`FakeEngine`), so tests run on any machine with no GPU
and no PaddleOCR model download.

---

## حدود الطبقة (مهم)

| يفعل | لا يفعل |
|---|---|
| قراءة PDF/صورة/DOCX/PPTX/XLSX من مسار | استقبال رفع HTTP مباشرة |
| OCR تكيّفي + جداول + تخطيط + توقيع/ختم | تصنيف نوع الوثيقة أو استخراج حقول عمل |
| Vision fallback عند الحاجة فقط | تشغيل Qwen-VL على كل صفحة |
| إرجاع `ocr_result` (العقد أعلاه) في state | كتابة نتائج إلى Storage |
| OCR للصور داخل PDF/الصور | OCR للصور المُضمّنة داخل DOCX/PPTX/XLSX (نص فقط حالياً) |
