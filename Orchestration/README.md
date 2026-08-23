# Orchestration Layer

**اللغة:** Python 3.11+ · **الإطار:** FastAPI (نقطة الدخول) + محرك Workflow داخلي (بدون LangGraph أو أي مكتبة graph خارجية)
**المسؤولية:** استقبال مهمة المعالجة من Application، تهيئة الحالة المركزية، تشغيل الـ Workflow (Supervisor → Agent → Result → State Update → Decision → Next Agent)، وإرجاع العقد النهائي `{ Success, Data }`.

الاتصال مع Agents هو **استدعاء دالة داخل العملية** (`agent.run(state)`)، وليس عبر الشبكة، وليس Agent-to-Agent مباشرة. كل استدعاء يمر عبر: **Orchestration → Agent → Result → State Update → Decision → Next Agent**.

> **حالة تكامل الـ Agents اليوم:** فقط `OCRAgent` مربوط فعليًا. باقي المراحل (Classification, Extraction, Validation, RAG, Summary, Routing, Writing) معرّفة بالكامل في الـ Graph والـ Config لكنها `enabled: false` حتى يتم ربطها واحدة تلو الأخرى (انظر "نقاط التكامل المستقبلية" أسفل).

---

## التدفق الكامل

```text
Request
  → StateManager.initialize()      (تهيئة الحالة المركزية GraphState)
  → SupervisorAgent.decide_before  (هل المرحلة متاحة/مفعّلة؟)
  → Agent adapter .run(state)      (تنفيذ الـ Agent، داخل العملية)
  → StateManager.apply_result      (كتابة نتيجة المرحلة في قسمها فقط)
  → SupervisorAgent.decide_after   (استمرار / إعادة محاولة / fallback / تخطي / إنهاء)
  → ... حتى النهاية أو الإنهاء المبكر
  → Final State + { Success, Data }
```

التسلسل الافتراضي للمراحل (`graph/graph_definition.py`):

```text
OCR → Classification → Extraction → Validation → RAG → Summary → Routing → Writing
```

الانتقالات ليست ثابتة بالضرورة: `supervisor/routing_logic.py` قد يتخطى RAG إذا لم تتطلبه نتيجة Validation، ويعيد المحاولة أو يسقط إلى fallback أو ينهي الـ Workflow حسب `config.yaml`.

### نقطة الدخول في `process_service.py`

| الدالة | الوصف |
|---|---|
| `run_workflow(...)` | المسار الوحيد: تشغيل Graph المراحل (OCR → … → Writing). المراحل غير المفعّلة تُتخطى. اليوم OCR فقط مفعّل، فيعمل كمرحلة أولى مثل أي Agent. |

`POST /process` في `main.py` يستدعي `run_workflow` فقط (لا مسار OCR منفصل، ولا `/process/full`).

`run_full_workflow` يبقى اسماً مرادفاً للتوافق مع الاستيرادات القديمة.

---

## تشغيل الطبقة

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

المنفذ **8000**. تحقق: `http://127.0.0.1:8000/health` → `{"status":"ok"}`

## تشغيل الاختبارات

```bash
pip install -r Orchestration/requirements.txt pytest
python -m pytest Orchestration/tests -v
```

31 اختبار، تغطي: تهيئة/تحديث الحالة، تنفيذ الـ Agent (نجاح/فشل/استثناء/نتيجة غير صالحة)، الانتقالات، قرارات الـ Supervisor، إعادة المحاولة/fallback/الإنهاء، الـ Workflow الكامل (mocked)، عقد الحالة النهائي، والاستيراد/عدم وجود circular imports. كل الـ Mocks داخل `Orchestration/tests/` فقط، ولا تُستخدم كتطبيقات Agent حقيقية.

---

## الإعدادات (`config.yaml`)

لكل مرحلة: `enabled`, `module`, `class_name` (المسار للاستيراد الكسول عند التفعيل فقط), `retries`, `timeout_seconds`, `fallback` (`terminate` | `skip` | `fallback_stage`). ملف فارغ/غائب = القيم الافتراضية البرمجية (OCR فقط مفعّل). `supervisor.max_total_retries` هو حد أمان لكامل الـ Workflow.

---

## معالجة الأخطاء

كل تنفيذ Agent يُعاد كأحد: `success | failure | invalid_result | missing_state | exception | skipped | not_integrated` (`messages/message_schema.py::ExecutionStatus`). لا يوجد ابتلاع صامت للأخطاء: كل حالة غير ناجحة تُسجَّل في `state["errors"]` وفي الـ logs (بدون محتوى المستند/الرسائل الحسّاسة - راجع `StateManager.snapshot`).

---

## وظيفة كل ملف

| الملف | الوظيفة |
|---|---|
| `main.py` | نقطة الدخول. FastAPI: `GET /health`, `POST /process` (الـ Workflow الكامل). |
| `process_service.py` | `run_workflow` بدون FastAPI؛ OCR مرحلة داخل الـ Graph مثل بقية الـ Agents. |
| `__init__.py` | يجعل المجلد حزمة Python قابلة للاستيراد. |
| `requirements.txt` | fastapi/uvicorn/pydantic/pyyaml. |
| `README.md` | هذا الملف. |
| `config.yaml` | سياسات الـ Workflow: enabled/retries/timeouts/fallback لكل مرحلة. |

### `messages/`

| الملف | الوظيفة |
|---|---|
| `message_schema.py` | عقد `{ Success, Data }` (تمرير من `Agents.ocr_agent.models` مع fallback آمن للاختبار المعزول) + عقود داخلية: `AgentRequest`, `AgentResult`, `ExecutionStatus`. |

### `state/`

| الملف | الوظيفة |
|---|---|
| `graph_state.py` | `GraphState`: الحالة المركزية الموحّدة (request/ocr/classification/extraction/validation/rag/summary/routing/writing + تحكم الـ Workflow). متوافقة خلفيًا مع مفاتيح OCR الأصلية. |
| `state_manager.py` | **نقطة الكتابة الوحيدة** للحالة: تهيئة، تطبيق نتيجة مرحلة في قسمها فقط، تتبّع المحاولات/التاريخ، تسجيل الأخطاء المُهيكلة، snapshot آمن للـ logging. |

### `supervisor/`

| الملف | الوظيفة |
|---|---|
| `supervisor_agent.py` | العقل المركزي: يقرر بدء/استمرار كل مرحلة عبر `routing_logic`. الوضع الحالي `deterministic` فقط. |
| `routing_logic.py` | قواعد انتقال حتمية: تخطٍّ/إعادة محاولة/fallback/إنهاء/اكتمال، بما فيها فروع عملية (تخطي RAG عند عدم الحاجة). |
| `supervisor_prompts.py` | قوالب Prompts محجوزة لوضع Supervisor مستقبلي مبني على LLM (`supervisor.mode: llm`) - غير مستخدمة حاليًا. |

### `workflow/`

| الملف | الوظيفة |
|---|---|
| `workflow_builder.py` | Adapter موحّد لأي Agent يطابق `run(state) -> state`، سجلّ الـ Agents (استيراد كسول حسب `config.yaml`، أو `agent_overrides` للاختبار)، ومحرك التنفيذ `Workflow.run(...)`. |
| `workflow_config.py` | تحميل `config.yaml` إلى إعدادات مكتوبة (typed) مع قيم افتراضية آمنة. |

### `graph/`

| الملف | الوظيفة |
|---|---|
| `graph_definition.py` | `Stage` enum، وصف كل عقدة (`GraphNode`)، والتسلسل الافتراضي `DEFAULT_SEQUENCE` + الحواف الخطية. لا تبعية لأي مكتبة graph خارجية. |

### `tests/`

اختبارات Orchestration المعزولة، مع `mock_agents.py` (Mocks للاختبار فقط، لا تُستخدم كتطبيق Agent حقيقي).

---

## نقاط التكامل المستقبلية (لكل Agent قادم)

لربط Agent حقيقي جديد (مثال: `classification_agent`):

1. تأكد أن الـ Agent يوفّر `run(self, state: dict) -> dict` ويكتب فقط `classification_result` (و`classification_status` اختياريًا) في الحالة التي يستقبلها.
2. في `config.yaml`: عيّن `agents.classification.enabled: true`، وتأكد أن `module`/`class_name` يشيران للمسار الصحيح.
3. لا حاجة لتعديل أي كود Orchestration آخر - `workflow_builder.build_agent_registry` سيستورد الـ Agent كسولًا عند أول تشغيل.
4. أضف اختبارات في `Orchestration/tests/` باستخدام mock مطابق للواجهة الحقيقية قبل التفعيل في `config.yaml`، ثم فعّله.

نفس الخطوات لبقية المراحل: `extraction`, `validation`, `rag`, `summary`, `routing` (routing_agent - وجهة العمل/القسم، مختلف عن Orchestration routing الذي يقرر "أي Agent التالي")، و`writing`.
