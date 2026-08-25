# SmartGovernmentAI

منصة **Smart Government Document AI** لتحليل الوثائق والمراسلات الرسمية: رفع وثيقة، معالجتها عبر Multi-Agent workflow، الاستناد إلى المعرفة القانونية (RAG)، ثم إرجاع نتيجة التحليل والتوجيه.

المشروع منظم كـ **Layered Architecture** (مجلد لكل طبقة داخل نفس المستودع)، وليس Microservices منفصلة.

للتفاصيل المعمارية راجع: [`Documentation/architecture.md`](Documentation/architecture.md)

---

## Project Structure

```
SmartGovernmentAI/
├── Presentation/          # واجهة المستخدم (React / TypeScript)
├── Application/           # واجهة API وخدمات الأعمال (C++)
├── Orchestration/         # إدارة سير العمل / Supervisor (Python)
├── Agents/                # وكلاء المعالجة المستقلون (Python)
├── Inference/             # استدلال النموذج اللغوي (llama.cpp + Python client)
├── RAG/                   # استرجاع المعرفة القانونية (Python · جاهزة للتشغيل)
├── Optimization/          # تصنيف أولي سريع (ONNX)
├── Storage/               # تخزين علائقي + ملفات (Python + SQL)
│
├── Config/                # إعدادات مشتركة
├── Documentation/         # وثائق معمارية ومخططات
├── Tests/                 # اختبارات حسب الطبقة
├── Docker/                # Dockerfiles للطبقات
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## الطبقات — الوظيفة

| الطبقة | التقنية | المسؤولية |
|---|---|---|
| **Presentation** | React / TypeScript | صفحات الرفع/النتائج/لوحة التحكم؛ تستدعي API عبر `services/` فقط |
| **Application** | C++ (`Controllers` / `Services` / `Routes` / `DTOs`) | استقبال الطلبات، التحقق، التفويض إلى Orchestration، قراءة الحالة |
| **Orchestration** | Python (`supervisor` / `workflow` / `graph` / `state`) | Supervisor يختار وينسّق الـ Agents ويحدّث حالة المعالجة |
| **Agents** | Python (`base` + وكلاء مستقلة) | OCR، تصنيف، استخراج، تحقق، RAG، تلخيص، صياغة، توجيه |
| **Inference** | llama-server + `client/llama_client.py` | توليد/استنتاج للنموذج؛ Agents تستدعي الـ client فقط |
| **RAG** | LangChain · ChromaDB · BGE-M3 · BM25 | فهرسة واسترجاع النصوص القانونية (Hybrid + PRF + Rerank) |
| **Optimization** | ONNX Runtime | تصنيف/فلترة سريعة قبل استدعاء LLM كامل |
| **Storage** | Python repositories + SQL migrations | وثائق، مستخدمون، حالة المعالجة، نتائج الـ Agents، الملفات |

طبقات مساعدة:

| المجلد | الوظيفة |
|---|---|
| `Config/` | إعدادات عامة وتسجيل وأمثلة أسرار |
| `Documentation/` | Architecture، API، كتالوج Agents، ADRs |
| `Tests/` | اختبارات مرآة للطبقات (الأكثر اكتمالاً حالياً: `Tests/RAG`) |
| `Docker/` | `Dockerfile.*` لكل طبقة رئيسية |

---

## العلاقة بين الطبقات وتدفق الطلب

التدفق المستهدف في المشروع:

```
Presentation  --REST-->  Application
                              │
                              ├── يكتب/يقرأ عبر Storage (Repositories)
                              └── يطلق معالجة ──► Orchestration
                                                      │
                                                      ▼
                                                   Agents
                                                      │
                              ┌───────────────────────┼───────────────────────┐
                              ▼                       ▼                       ▼
                         Inference                  RAG                 Optimization
                              │
                              └── النتائج تعود عبر Orchestration state
                                  ثم تُحفظ في Storage وتُعرض من Presentation
```

ملاحظات مهمة عن الواقع الحالي:

- **Presentation → Application:** عبر REST (`Presentation/services` → Application Controllers).
- **Application → Storage / Orchestration:** Application تحتوي Controllers/Services؛ Storage منفصل بمستودعات Python وملفات SQL.
- **Orchestration ↔ Agents:** استدعاء داخل نفس العملية (In-Process)، وليس شبكة.
- **Agents → Inference / RAG / Optimization:** عبر واجهات client/خدمة مشتركة؛ الـ Agents لا تكتب إلى Storage مباشرة.
- طبقة **RAG** هي الأكثر اكتمالاً وتشغيلاً حالياً؛ باقي الطبقات موجودة هيكلياً بدرجات نضج متفاوتة.

---

## التقنيات المستخدمة فعلياً

| المجال | الموجود في المستودع |
|---|---|
| Frontend | React، TypeScript (`Presentation/`) |
| API | C++ Controllers/Services/Routes (`Application/`) |
| Orchestration | Python · هيكل LangGraph/Supervisor (`Orchestration/`) |
| Agents | Python · `BaseAgent` + registry |
| LLM Inference | llama.cpp server launchers + Python HTTP client |
| RAG | LangChain، ChromaDB، HuggingFace Embeddings (`BAAI/bge-m3`)، BM25، CrossEncoder |
| Optimization | ONNX Runtime wrappers |
| Storage | SQL migrations + Python repositories/models |
| Ops | Dockerfiles، `.env.example`، `Config/*.yaml` |

---

## طريقة التشغيل (حسب ما هو واضح حالياً)

### المتطلبات العامة
- Python 3.11+ (لـ RAG / Orchestration / Agents / Storage)
- Node.js (عند تشغيل Presentation بعد إعداد اعتماديات الواجهة)
- أداة بناء C++ / CMake (لـ Application)
- اختياري: Docker حسب ملفات `Docker/`

### 1) إعداد البيئة
```bash
cp .env.example .env
# راجع أيضاً: Config/secrets.env.example و Config/global_config.yaml
```

> ملاحظة: `docker-compose.yml` الحالي تعليقات فقط؛ لا تعتمد عليه لتشغيل كامل المنظومة بعد.

### 2) RAG (الطبقة الجاهزة للتشغيل محلياً)

ضع ملفاتك في:
- `RAG/documents/laws/`
- `RAG/documents/regulations/`
- `RAG/documents/internal_docs/`
- `RAG/documents/uploads/`

ثم:
```bash
pip install -r RAG/requirements.txt
python -m RAG.ingestion.pipeline --reset
python Tests/RAG/test_pipeline.py
python RAG/scripts/smoke_check.py --fast
```

تفاصيل إضافية: [`RAG/README.md`](RAG/README.md)

### 3) Inference (اختبار العميل)
```bash
# شغّل llama-server أولاً عبر Inference/llama_server/
python Tests/Inference/test_llama_client.py
```

### 4) Presentation / Application / Orchestration
الهيكل والـ entry points موجودة (`Presentation/main.tsx`، `Application/main.cpp`، `Orchestration/main.py`)، لكن التشغيل الكامل end-to-end يعتمد على إكمال الإعداد والربط بين الطبقات. ابدأ من README داخل كل مجلد ومن `Documentation/`.

---

## للمستجد في المشروع — من أين تبدأ؟

1. اقرأ هذا الملف ثم [`Documentation/architecture.md`](Documentation/architecture.md)
2. افهم التدفق: Presentation → Application → Orchestration → Agents → (Inference / RAG / Optimization) + Storage
3. جرّب طبقة **RAG** أولاً لأنها الطبقة الأكثر قابلية للتشغيل والاختبار الآن
4. راجع وكلاء المعالجة تحت `Agents/` وواجهات الاستدعاء في `Inference/client/` و `RAG/client/`

---

## روابط سريعة

| الموضوع | المسار |
|---|---|
| Architecture | `Documentation/architecture.md` |
| مخطط الربط | `Documentation/layer_connections_diagram.svg` |
| كتالوج Agents | `Documentation/agent_catalog.md` |
| مرجع API | `Documentation/api_reference.md` |
| دليل RAG | `RAG/README.md` |


PaddleOCR-VL — المنفذ 8111
cd D:\AI\KutupAI\Inference
.\start_paddleocr_vl.bat
انتظر حتى يظهر أن السيرفر جاهز على http://127.0.0.1:8111

2) Gemma (Inference) — المنفذ 8080
للتصنيف / الاستخراج / الملخص / الكتابة:

cd D:\AI\KutupAI\Inference\llama_server
.\server_launcher.bat
انتظر حتى يصبح جاهزاً على :8080

3) Orchestration — المنفذ 8000
cd D:\AI\KutupAI
$env:ORCHESTRATION_PORT="8000"
python -m Orchestration.main
تحقق: http://127.0.0.1:8000/health

4) Application — المنفذ 8082
(لا تستخدم 8080 — محجوز لـ Gemma)

أولاً إن لزم أعد البناء (أوقف أي SmartGovernmentAI_Application.exe شغّال):

cd D:\AI\KutupAI\Application
cmake --build build --config Release --target SmartGovernmentAI_Application
ثم التشغيل:

cd D:\AI\KutupAI\Application
$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="D:\AI\KutupAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8082"
.\build\Release\SmartGovernmentAI_Application.exe
5) الواجهة — المنفذ 5173
cd D:\AI\KutupAI\Tests\Presentation
npm run dev
افتح: http://localhost:5173

ملخص المنافذ
الخدمة	المنفذ
PaddleOCR-VL
8111
Gemma (llama)
8080
Orchestration
8000
Application
8082
Presentation
5173
بعد ما الخمسة يشتغلوا: ارفع الملف من الواجهة + اكتب سؤال → أرسل.

cd D:\AI\KutupAI
python -m RAG.ingestion.pipeline --reset