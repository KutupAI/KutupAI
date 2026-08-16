# Orchestration Layer

**اللغة:** Python · **الإطار:** FastAPI (نقطة الدخول) + بنية Supervisor/LangGraph للخطوات اللاحقة  
**المسؤولية:** استقبال مهمة المعالجة من Application، تهيئة الحالة، استدعاء Worker Agents داخل نفس العملية، وإرجاع العقد النهائي `{ Success, Data }`.

الاتصال مع Agents هو **استدعاء دالة داخل العملية** (ليس شبكة). Orchestration فقط يكتب حالة/نتائج الـ Workflow إلى Storage لاحقًا؛ الـ Agents لا تتصل بـ Storage.

حاليًا خطوة OCR فقط: `run_ocr_pipeline` → `OCRAgent.run`.

---

## التدفق

```text
Application  POST /process
  { document_id, document_path, question }
      ↓
main.py
      ↓
process_service.run_ocr_pipeline
      ↓
Agents/ocr_agent OCRAgent.run(state)     (in-process)
      ↓
إرجاع { Success, Data: [ document ] }
```

`Success = true` إذا اكتملت معالجة الوثيقة. `false` إذا المسار ناقص أو OCR فشل تمامًا.

---

## تشغيل الطبقة

المتطلبات في `requirements.txt` (هذه الطبقة) **و** `Agents/ocr_agent/requirements.txt` لأن الاستدعاء داخل العملية.

```powershell
cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI
pip install -r Orchestration\requirements.txt
pip install -r Agents\ocr_agent\requirements.txt

$env:ORCHESTRATION_PORT="8000"
$env:QWEN_VL_ENABLED="true"
$env:QWEN_VL_ENDPOINT="http://127.0.0.1:8081/v1/chat/completions"
$env:QWEN_VL_TIMEOUT="240"
python -m Orchestration.main
```

المنفذ **8000**. يحتاج ملفًا على قرص مشترك كتبه Application، وخادم Qwen-VL على **8081** إن لزم الاسترداد البصري.

تحقق: `http://127.0.0.1:8000/health` → `{"status":"ok"}`

---

## وظيفة كل ملف

| الملف | الوظيفة |
|---|---|
| `main.py` | نقطة الدخول. FastAPI: `GET /health` و `POST /process`. يحوّل طلب Application إلى `run_ocr_pipeline`. |
| `process_service.py` | تشغيل OCR بدون FastAPI (للاختبارات). يبني الحالة، يستدعي `OCRAgent`، يثبّت عقد `{ Success, Data }`. |
| `__init__.py` | يجعل المجلد حزمة Python قابلة للاستيراد. |
| `requirements.txt` | حزم FastAPI/Uvicorn. |
| `README.md` | هذا الملف. |
| `config.yaml` | إن وُجد: إعدادات تشغيل إضافية. |

### `messages/`

| الملف | الوظيفة |
|---|---|
| `message_schema.py` | عقد الرسائل الموحّد `{ Success, Data }` على جانب Orchestration (يمرّر نفس الـ schema من OCR Agent). |

### `state/`

| الملف | الوظيفة |
|---|---|
| `graph_state.py` | الحالة المشتركة بين Supervisor والـ Agents: `document_id`, `document_path`, `question`, `ocr_result` (`{ Success, Data }`), `ocr_status`. |
| `state_manager.py` | هيكل حفظ/استرجاع checkpoint الحالة إلى Storage. |

### `supervisor/`

| الملف | الوظيفة |
|---|---|
| `supervisor_agent.py` | هيكل العقل المركزي: اختيار الـ Agent التالي من `agent_registry`. |
| `routing_logic.py` | هيكل قواعد الانتقال (مثال: بعد OCR → Classification إلا إذا فشل OCR). |
| `supervisor_prompts.py` | هيكل قوالب Prompts للـ Supervisor. |

### `workflow/`

| الملف | الوظيفة |
|---|---|
| `workflow_builder.py` | هيكل بناء LangGraph: أي Node يمثل أي Agent. |
| `workflow_config.py` | هيكل إعدادات الـ Graph. |

### `graph/`

| الملف | الوظيفة |
|---|---|
| `graph_definition.py` | هيكل تعريف StateGraph والـ Edges الشرطية. |
