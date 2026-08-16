# Smart Government Document AI System
## Production Project Structure — Software Architecture Document

**النوع:** Layered Architecture (بدون Microservices)
**الحالة:** Structure Only — لا يوجد أي كود تنفيذي في هذه الوثيقة

---

## جدول المحتويات

1. [نظرة عامة](#نظرة-عامة)
2. [المبادئ المعمارية](#المبادئ-المعمارية)
3. [البنية الجذرية للمشروع](#البنية-الجذرية-للمشروع-root-structure)
4. [الطبقات بالتفصيل](#الطبقات-بالتفصيل)
   - [1) Presentation Layer](#1-presentation-layer)
   - [2) Application Layer](#2-application-layer)
   - [3) Orchestration Layer](#3-orchestration-layer)
   - [4) Worker Agents Layer](#4-worker-agents-layer)
   - [5) AI Inference Layer](#5-ai-inference-layer)
   - [6) Knowledge / RAG Layer](#6-knowledge--rag-layer)
   - [7) Optimization Layer](#7-optimization-layer)
   - [8) Data Storage Layer](#8-data-storage-layer)
5. [الربط بين الطبقات (Data Flow)](#الربط-بين-الطبقات-layer-connections--data-flow)
6. [قابلية التوسّع](#قابلية-التوسع-extensibility)
7. [Tests / Docker / Documentation / Config](#tests--docker--documentation--config)

---

## نظرة عامة

نظام **Smart Government Document AI System** هو منصة Multi-Agent AI تستقبل الوثائق والمراسلات الرسمية الحكومية التركية، تحلّلها، تصنّفها، تستخرج البيانات منها، تبحث في القوانين والتشريعات ذات الصلة (Mevzuat)، تصيغ ردًا/كتابًا رسميًا، ثم توجّه الوثيقة إلى القسم المختص — كل ذلك عبر مجموعة Agents مستقلة يديرها Supervisor Agent ضمن Orchestration Layer.

---

## المبادئ المعمارية

| # | المبدأ | كيف يتحقق في هذه البنية |
|---|--------|--------------------------|
| 1 | Layered Architecture فقط — لا Microservices | كل طبقة مجلد مستقل، لا شبكة خدمات منفصلة؛ التواصل بين Python Layers داخل نفس الـ Process |
| 2 | مسؤولية واضحة لكل طبقة | كل طبقة أدناه لها قسم "المسؤولية" منفصل ولا تتقاطع مع غيرها |
| 3 | Agents مستقلة وليست Pipeline ثابتة | نمط `base_agent.py` + `agent_registry.py` يسمح للـ Supervisor باختيار Agent ديناميكيًا |
| 4 | فصل Business Logic عن AI Logic | Application Layer = منطق عمل/API فقط؛ كل ما هو AI في Orchestration/Agents/Inference/RAG/Optimization |
| 5 | استبدال Gemma 3 لاحقًا دون كسر النظام | Agents تستدعي `Inference/client/` فقط (واجهة ثابتة)، لا تعرف تفاصيل النموذج |
| 6 | استبدال Vector Database لاحقًا | `RAG/vector_store/vector_store_interface.py` تجريد يُخفي ChromaDB عن باقي الطبقات |
| 7 | نظام قابل لإضافة Agents جديدة | إضافة مجلد جديد + تسجيله في `agent_registry.py` — بدون تعديل Orchestration Core |
| 8 | نقطة كتابة مركزية واحدة لقاعدة البيانات | Storage يُكتب فقط من Application (بيانات API) و Orchestration (حالة/نتائج)؛ الـ Agents لا تتصل بـ Storage مباشرة |

---

## البنية الجذرية للمشروع (Root Structure)

```
SmartGovernmentAI/
│
├── Presentation/                 # TypeScript · React
├── Application/                  # C++ · Drogon/Crow
├── Orchestration/                # Python · LangGraph
├── Agents/                       # Python · Worker Agents (مستقلة)
├── Inference/                    # C/C++ · llama.cpp
├── RAG/                          # Python · BGE-M3 + ChromaDB
├── Optimization/                 # Python/C++ · ONNX Runtime
├── Storage/                      # SQL · PostgreSQL/SQL Server
│
├── Config/                       # إعدادات مشتركة بين كل الطبقات
├── Documentation/                # وثائق معمارية + مخططات
├── Tests/                        # اختبارات مرآة لكل طبقة
├── Docker/                       # Dockerfiles لكل طبقة + docker-compose
│
├── docker-compose.yml            # تشغيل كل الطبقات محليًا كوحدة واحدة
├── .env.example                  # قالب متغيرات البيئة (بدون قيم حقيقية)
├── .gitignore
└── README.md
```

| العنصر | المسؤولية |
|---|---|
| `docker-compose.yml` | تشغيل Presentation + Application + Orchestration(+Agents) + Inference + RAG + قاعدة البيانات كوحدة واحدة للتطوير المحلي |
| `.env.example` | توثيق كل متغيرات البيئة المطلوبة (منافذ، مسارات النماذج، سلاسل اتصال) دون كشف قيم حساسة |
| `README.md` | نظرة عامة، طريقة التشغيل، روابط لبقية التوثيق في `Documentation/` |

---

## الطبقات بالتفصيل

### 1) Presentation Layer

**اللغة:** TypeScript **· الإطار:** React

```
Presentation/
├── components/
│   ├── DocumentUploader/
│   │   ├── DocumentUploader.tsx
│   │   └── DocumentUploader.module.css
│   ├── AnalysisResultCard/
│   │   └── AnalysisResultCard.tsx
│   ├── StatusBadge/
│   │   └── StatusBadge.tsx
│   └── Layout/
│       ├── Header.tsx
│       ├── Sidebar.tsx
│       └── Footer.tsx
│
├── pages/
│   ├── UploadPage.tsx
│   ├── ResultsPage.tsx
│   ├── DashboardPage.tsx
│   └── LoginPage.tsx
│
├── services/
│   ├── apiClient.ts
│   ├── documentService.ts
│   ├── authService.ts
│   └── statusService.ts
│
├── hooks/
│   ├── useDocumentUpload.ts
│   ├── useProcessingStatus.ts
│   └── useAuth.ts
│
├── models/
│   ├── Document.ts
│   ├── AnalysisResult.ts
│   ├── User.ts
│   └── ApiResponse.ts
│
├── store/
│   ├── documentStore.ts
│   ├── authStore.ts
│   └── index.ts
│
├── utils/
│   ├── formatters.ts
│   ├── validators.ts
│   └── constants.ts
│
├── config/
│   ├── apiConfig.ts
│   └── appConfig.ts
│
├── App.tsx
├── main.tsx
├── package.json
├── tsconfig.json
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `components/DocumentUploader.tsx` | واجهة رفع الوثيقة (Drag & Drop/Browse)، يستدعي `documentService` |
| `components/AnalysisResultCard.tsx` | عرض نتيجة التحليل (تصنيف، بيانات مستخرجة، الجهة المقترحة للتوجيه) |
| `components/StatusBadge.tsx` | عرض حالة المعالجة (Pending / Processing / Done / Failed) |
| `pages/UploadPage.tsx` | صفحة رفع وثيقة جديدة |
| `pages/ResultsPage.tsx` | صفحة عرض نتيجة تحليل وثيقة محددة |
| `pages/DashboardPage.tsx` | لوحة متابعة عامة لكل الوثائق وحالاتها |
| `services/apiClient.ts` | Wrapper موحّد فوق fetch/axios (Headers, Auth Token, Error Handling) — كل الخدمات تمر من هنا |
| `services/documentService.ts` | استدعاءات API الخاصة بالوثائق: `upload()`, `getResult()`, `getStatus()` |
| `services/authService.ts` | تسجيل الدخول وتجديد التوكن |
| `hooks/useDocumentUpload.ts` | منطق حالة الرفع (progress, error, success) قابل لإعادة الاستخدام |
| `hooks/useProcessingStatus.ts` | Polling دوري لحالة المعالجة من Application API |
| `models/*.ts` | تعريفات TypeScript تُطابق تمامًا DTOs في Application Layer |
| `store/` | إدارة الحالة العامة (المستخدم الحالي، الوثيقة الجارية) |
| `utils/` | دوال مساعدة عامة، بلا اعتماد على React |
| `config/apiConfig.ts` | Base URL الخاص بـ Application API ومهلات الاتصال |

**ملاحظة مهمة:** طبقة `services/` هي المكان الوحيد المسموح فيه بالاتصال بالخارج؛ لا تستدعي أي Component أو Page الـ API مباشرة.

---

### 2) Application Layer

**اللغة:** C++ **· الإطار:** Drogon أو Crow

```
Application/
├── Controllers/
│   ├── DocumentController.h / .cpp
│   ├── AuthController.h / .cpp
│   ├── UserController.h / .cpp
│   └── StatusController.h / .cpp
│
├── Routes/
│   ├── DocumentRoutes.cpp
│   ├── AuthRoutes.cpp
│   └── ApiRoutes.h            # تجميع كل الـ Routes
│
├── DTOs/
│   ├── DocumentRequestDTO.h
│   ├── DocumentResponseDTO.h
│   ├── AuthDTO.h
│   └── UserDTO.h
│
├── Services/
│   ├── DocumentProcessingService.h / .cpp
│   ├── AuthService.h / .cpp
│   └── UserService.h / .cpp
│
├── Middleware/
│   ├── AuthMiddleware.h / .cpp
│   ├── LoggingMiddleware.h / .cpp
│   ├── ErrorHandlingMiddleware.h / .cpp
│   └── CorsMiddleware.h / .cpp
│
├── Validators/
│   ├── DocumentValidator.h / .cpp
│   └── RequestValidator.h / .cpp
│
├── Configuration/
│   ├── AppConfig.h / .cpp
│   └── DatabaseConfig.h
│
├── main.cpp
├── CMakeLists.txt
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `Controllers/DocumentController` | استقبال طلبات رفع/استعلام الوثائق، يستدعي `DocumentProcessingService` |
| `Controllers/AuthController` | تسجيل الدخول، إصدار وتجديد التوكن |
| `Controllers/StatusController` | إرجاع حالة معالجة وثيقة معيّنة (قراءة من Storage) |
| `Routes/*Routes.cpp` | ربط كل Endpoint بدالة Controller المطابقة وفق `api/[controller]/[action]` |
| `DTOs/*` | أشكال الطلب/الاستجابة الخارجية — لا تُستخدم داخليًا في باقي الطبقات |
| `Services/DocumentProcessingService` | الجسر الوحيد بين Application و Orchestration؛ يحفظ الوثيقة في Storage ثم يطلق Job في Orchestration |
| `Services/AuthService` | منطق التحقق من الهوية وإصدار الجلسات |
| `Middleware/AuthMiddleware` | التحقق من صلاحية التوكن قبل الوصول للـ Controllers |
| `Middleware/LoggingMiddleware` | تسجيل كل طلب/استجابة (مرتبط بـ Storage/Logs) |
| `Validators/DocumentValidator` | التحقق من صيغة/حجم/نوع الملف قبل المعالجة |
| `Configuration/AppConfig` | إعدادات التشغيل العامة (منفذ الخادم، مهلات، عناوين الخدمات الداخلية) |

**قاعدة صلبة:** هذه الطبقة **لا تحتوي أي منطق AI** — فقط استقبال، تحقق، تفويض إلى Orchestration، وإرجاع نتائج جاهزة.

---

### 3) Orchestration Layer

**اللغة:** Python **· الإطار:** LangGraph (Supervisor Agent)

```
Orchestration/
├── workflow/
│   ├── workflow_builder.py     # بناء LangGraph StateGraph
│   └── workflow_config.py
│
├── supervisor/
│   ├── supervisor_agent.py     # منطق اختيار الـ Agent التالي
│   ├── routing_logic.py
│   └── supervisor_prompts.py
│
├── state/
│   ├── graph_state.py          # تعريف الحالة المشتركة (TypedDict) بين كل الـ Nodes
│   └── state_manager.py        # حفظ/استرجاع نقاط checkpoint إلى Storage
│
├── messages/
│   └── message_schema.py       # شكل الرسائل الداخلية بين Supervisor والـ Agents
│
├── graph/
│   └── graph_definition.py     # تعريف الـ Nodes والـ Edges الشرطية
│
├── main.py                     # نقطة الدخول (خدمة داخلية تستقبل الطلب من Application)
├── requirements.txt
├── config.yaml
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `workflow/workflow_builder.py` | يبني الـ Graph الكامل: أي Node يمثل أي Agent |
| `supervisor/supervisor_agent.py` | **العقل المركزي**: يقرأ الحالة الحالية، يستشير `agent_registry` في Agents Layer، يختار الـ Agent المناسب |
| `supervisor/routing_logic.py` | قواعد الانتقال بين الـ Agents (مثال: بعد OCR → Classification، إلا إذا فشل OCR) |
| `state/graph_state.py` | الحالة المشتركة: نص الوثيقة، النتائج الجزئية لكل Agent، القرار النهائي |
| `state/state_manager.py` | **نقطة الكتابة الوحيدة** لحالة الـ Workflow ونتائج الـ Agents في Storage |
| `messages/message_schema.py` | تنسيق موحّد لتبادل الرسائل بين Supervisor والـ Agents |
| `graph/graph_definition.py` | تعريف LangGraph الفعلي (StateGraph, conditional_edges) |
| `main.py` | يعرض Endpoint داخلي بسيط (مثال: `POST /process`) يستدعيه Application Layer |

**مهم:** الاتصال بين Orchestration و Agents هو **استدعاء دالة داخل نفس العملية (In-Process Call)** — ليس عبر الشبكة — لأن كليهما Python، وهذا يحافظ على مبدأ "Layered وليس Microservices".

---

### 4) Worker Agents Layer

**اللغة:** Python **· مبدأ:** Modules مستقلة، ليست Pipeline ثابتة

```
Agents/
├── base/
│   ├── base_agent.py            # Interface: run(state) -> state ينفذه كل Agent
│   └── agent_registry.py        # يسجّل كل Agent ليكتشفه Supervisor ديناميكيًا
│
├── ocr_agent/
│   ├── agent.py
│   ├── prompts.py
│   ├── tools.py
│   └── config.py
│
├── classification_agent/        (نفس بنية ocr_agent)
├── extraction_agent/            (نفس البنية)
├── validation_agent/            (نفس البنية)
├── rag_agent/                   (نفس البنية)
├── summary_agent/               (نفس البنية)
├── writer_agent/                (نفس البنية)
├── routing_agent/               (نفس البنية)
│
├── __init__.py
└── README.md
```

كل مجلد Agent يتبع نفس القالب الأربعي:

| الملف | المسؤولية |
|---|---|
| `agent.py` | الكلاس الرئيسي، يرث من `BaseAgent` وينفذ `run(state)` |
| `prompts.py` | قوالب الـ Prompts الخاصة بهذا الـ Agent (إن استخدم LLM) |
| `tools.py` | أدوات/تكاملات خارجية يحتاجها الـ Agent (مثال: PaddleOCR في ocr_agent) |
| `config.py` | إعدادات خاصة بالـ Agent (عتبات، مسارات نماذج مساعدة) |

| Agent | مسؤوليته الأساسية | يستدعي (Shared Services) |
|---|---|---|
| `ocr_agent` | تحويل صورة/PDF الوثيقة إلى نص (PaddleOCR) | — |
| `classification_agent` | تصنيف نوع الوثيقة | Optimization (تصنيف سريع أولي) ثم Inference عند الحاجة |
| `extraction_agent` | استخراج الحقول المهمة (رقم المعاملة، التاريخ، الجهة...) | Inference |
| `validation_agent` | التحقق من اكتمال وصحة البيانات المستخرجة وفق قواعد العمل | Inference (للحالات الغامضة) |
| `rag_agent` | البحث في القوانين/اللوائح ذات الصلة بمحتوى الوثيقة | RAG ثم Inference |
| `summary_agent` | تلخيص محتوى الوثيقة | Inference |
| `writer_agent` | صياغة الكتاب الرسمي النهائي | RAG (للاستناد القانوني) + Inference |
| `routing_agent` | تحديد القسم/الجهة المختصة لتوجيه الوثيقة إليها | Inference |

**قاعدة صلبة:** لا يوجد أي Agent يتصل بـ Storage مباشرة. كل Agent يُعيد نتيجته إلى `graph_state`، و`state_manager.py` في Orchestration هو من يحفظها.

---

### 5) AI Inference Layer

**اللغة:** C/C++

```
Inference/
├── llama_server/
│   ├── server_launcher.sh
│   └── build_config.cmake
│
├── models/
│   ├── gemma3.gguf              # (Binary — لا يُرفع لـ Git، مذكور هنا كمرجع فقط)
│   └── model_registry.json      # اسم النموذج، الإصدار، مستوى الـ Quantization
│
├── client/
│   ├── llama_client.py          # العميل الذي تستخدمه الـ Agents (Python)
│   ├── inference_request.py
│   └── inference_response.py
│
├── configuration/
│   ├── inference_config.yaml    # context size, threads, GPU layers
│   └── model_config.json
│
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `llama_server/` | بناء وتشغيل `llama-server` فوق `llama.cpp` |
| `models/model_registry.json` | معلومات وصفية عن النموذج الحالي — تُمكّن استبدال Gemma 3 لاحقًا دون تغيير الكود |
| `client/llama_client.py` | **الواجهة الوحيدة** التي تستخدمها Agents Layer؛ تُخفي تفاصيل البروتوكول عن llama-server |
| `configuration/inference_config.yaml` | ضبط الأداء (عدد الخيوط، طول السياق، طبقات GPU) |

**قابلية الاستبدال:** لاستبدال Gemma 3 بنموذج آخر لاحقًا، يكفي تغيير `models/` + `client/llama_client.py` الداخلي؛ الواجهة الخارجية (الدوال التي تستدعيها Agents) تبقى كما هي.

---

### 6) Knowledge / RAG Layer

**اللغة:** Python

```
RAG/
├── embeddings/
│   ├── embedding_model.py       # BGE-M3 wrapper
│   └── embedding_config.py
│
├── vector_store/
│   ├── vector_store_interface.py # تجريد عام (Abstract Interface)
│   └── chroma_store.py           # التطبيق الحالي فوق ChromaDB
│
├── chroma/
│   └── chroma_config.py          # مسار التخزين، إعدادات الـ Collection
│
├── retriever/
│   ├── retriever.py               # البحث الدلالي (Top-K)
│   └── reranker.py                # إعادة ترتيب النتائج (اختياري)
│
├── client/
│   ├── rag_client.py              # الواجهة الوحيدة التي تستخدمها Agents (مثل Inference/client)
│   ├── retrieval_request.py
│   └── retrieval_response.py
│
├── documents/
│   ├── laws/                      # Turkish Laws (Mevzuat)
│   ├── regulations/
│   └── internal_docs/
│
├── indexing/
│   ├── indexer.py                 # تقطيع + Embedding + إدخال للفهرس
│   └── update_index.py            # تحديث الفهرس دوريًا
│
├── configuration/
│   ├── rag_config.yaml            # إعدادات التشغيل (chunking / retrieval)
│   └── rag_config_loader.py
│
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `embeddings/embedding_model.py` | تحويل النص إلى متجهات باستخدام BGE-M3 |
| `vector_store/vector_store_interface.py` | العقد الذي يعتمد عليه باقي RAG Layer — **لا** ChromaDB مباشرة |
| `vector_store/chroma_store.py` | التطبيق الفعلي الحالي؛ يُستبدل لاحقًا دون المساس بـ `retriever.py` |
| `client/rag_client.py` | **الواجهة الوحيدة** التي تستخدمها `rag_agent` و `writer_agent` (نفس نمط `Inference/client`) |
| `retriever/retriever.py` | محرك البحث الدلالي الداخلي الذي يستدعيه `rag_client` |
| `documents/laws` | نصوص Mevzuat (القوانين التركية) المصدر |
| `indexing/indexer.py` | خط أنابيب الفهرسة: تقطيع → Embedding → حفظ في Vector Store |

**قابلية الاستبدال:** تغيير قاعدة المتجهات (مثلاً من ChromaDB إلى Qdrant) يتم فقط داخل `vector_store/`، دون التأثير على `retriever/` أو `embeddings/` أو أي Agent.

---

### 7) Optimization Layer

**اللغة:** Python / C++ **· التقنية:** ONNX Runtime

```
Optimization/
├── models/
│   ├── classification_model.onnx
│   └── model_metadata.json
│
├── runtime/
│   ├── onnx_runtime_wrapper.py
│   └── session_manager.py         # إدارة جلسات الاستدلال والـ Batching
│
├── services/
│   ├── fast_classification_service.py
│   └── preprocessing.py
│
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `models/classification_model.onnx` | نموذج خفيف مُحسَّن للتصنيف الأولي السريع |
| `runtime/session_manager.py` | إدارة جلسات ONNX (تحميل، إعادة استخدام، Batching) |
| `services/fast_classification_service.py` | الواجهة التي يستدعيها `classification_agent` قبل اللجوء إلى LLM كامل |

**الغرض:** تقليل الحمل على AI Inference Layer عبر فلترة/تصنيف أولي سريع ورخيص قبل استدعاء Gemma 3 عند الضرورة فقط.

---

### 8) Data Storage Layer

**قاعدة البيانات:** PostgreSQL أو SQL Server

```
Storage/
├── database/
│   ├── db_connection.py
│   └── db_config.yaml
│
├── repositories/
│   ├── document_repository.py
│   ├── user_repository.py
│   ├── processing_status_repository.py
│   ├── agent_result_repository.py
│   └── log_repository.py
│
├── migrations/
│   ├── 001_create_documents_table.sql
│   ├── 002_create_users_table.sql
│   ├── 003_create_processing_status_table.sql
│   └── 004_create_agent_results_table.sql
│
├── models/
│   ├── Document.py
│   ├── User.py
│   ├── ProcessingStatus.py
│   └── AgentResult.py
│
├── files/
│   ├── uploads/
│   └── processed/
│
├── schema.sql                     # المرجع الكامل للمخطط
└── README.md
```

| الملف / المجلد | المسؤولية |
|---|---|
| `repositories/document_repository.py` | كل عمليات CRUD الخاصة بالوثائق — **نقطة الوصول الوحيدة** لجدول Documents |
| `repositories/processing_status_repository.py` | تحديث/قراءة حالة المعالجة (يُستخدم من Application و Orchestration) |
| `repositories/agent_result_repository.py` | حفظ نتيجة كل Agent بعد اكتماله (يُستدعى من Orchestration فقط) |
| `migrations/*.sql` | تطور المخطط بمرور الوقت، بترتيب واضح وقابل للتتبع |
| `files/uploads` | تخزين الملفات الخام المرفوعة قبل المعالجة |

**قاعدة صلبة:** لا يستدعي أي Agent أي Repository مباشرة — الكتابة تمر حصرًا عبر Application (بيانات API) و Orchestration (حالة/نتائج).

---

## الربط بين الطبقات (Layer Connections & Data Flow)

المخطط المرفق **`layer_connections_diagram.svg`** يوضّح هذا الربط بصريًا. الفكرة الأساسية:

> **ليست كل الطبقات في سلسلة خطية واحدة.** هناك مسار تنفيذي رئيسي (Presentation → Application → Orchestration → Agents)، وهناك طبقات خدمة مشتركة (Inference / RAG / Optimization) تُستدعى من Agents *حسب الحاجة* فقط، وطبقة تخزين (Storage) تُكتب فقط من Application و Orchestration.

### أ) التدفق التسلسلي الرئيسي (خطوة بخطوة)

1. المستخدم يرفع وثيقة عبر **Presentation** (React).
2. Presentation تستدعي **Application** عبر REST API (HTTPS/JSON).
3. Application يتحقق من الهوية، يخزّن الوثيقة والحالة الأولية (`Received`) في **Storage**، ثم يطلق مهمة معالجة في **Orchestration** (استدعاء داخلي غير متزامن — Async Job Trigger).
4. Orchestration (Supervisor Agent) يهيّئ `graph_state` ويبدأ اختيار الـ Agents ديناميكيًا وفق `routing_logic.py`.
5. لكل Agent يتم استدعاؤه (استدعاء داخل نفس العملية، بلا شبكة):
   - قد يستدعي **Optimization** لتصنيف/فلترة سريعة.
   - قد يستدعي **AI Inference Layer** لتوليد نص أو استنتاج.
   - قد يستدعي **RAG Layer** لجلب نصوص قانونية ذات صلة.
6. نتيجة كل Agent تُعاد إلى `graph_state`؛ **Orchestration فقط** من يكتب هذه النتائج في **Storage** (`agent_result_repository`).
7. عندما ينتهي Supervisor من تنفيذ كل الـ Agents اللازمة (مثلاً بعد `routing_agent`)، تُرجَع النتيجة النهائية إلى **Application**.
8. Application يحدّث الحالة النهائية في Storage ويُعيد الاستجابة إلى Presentation (أو Presentation يستمر بعمل Polling على حالة المعالجة).
9. Presentation يعرض نتيجة التحليل، الكتاب المُصاغ، والجهة المقترحة للتوجيه.

### ب) مصفوفة الربط بين الطبقات

| من | إلى | آلية الاتصال | الغرض |
|---|---|---|---|
| Presentation | Application | REST API (HTTPS/JSON) | رفع وثيقة، استعلام نتيجة/حالة، مصادقة |
| Application | Storage | Repositories (اتصال مباشر) | حفظ الوثائق، المستخدمين، الحالة الأولية/النهائية |
| Application | Orchestration | استدعاء داخلي غير متزامن (Async Job Trigger) | إطلاق Workflow المعالجة |
| Orchestration | Agents | استدعاء دالة داخل نفس العملية (In-Process) | Supervisor يختار وينفّذ الـ Agent المناسب ديناميكيًا |
| Orchestration | Storage | Repositories (اتصال مباشر) | حفظ حالة الـ Workflow ونتائج كل Agent |
| Agents | AI Inference Layer | عميل داخلي مشترك (`llama_client`) | توليد نص/استنتاج لعدة Agents (Classification, Extraction, Validation, RAG, Summary, Writer, Routing) |
| Agents | RAG Layer | عميل داخلي (`RAG/client`) | استرجاع نصوص قانونية — يُستخدم أساسًا من `rag_agent` و `writer_agent` |
| Agents | Optimization Layer | عميل داخلي (`fast_classification_service`) | تصنيف/فلترة سريعة قبل اللجوء إلى LLM كامل — أساسًا `classification_agent` |
| Agents | Orchestration | إرجاع قيمة (تحديث `state`) | Agents لا تكتب لأي مكان مباشرة؛ فقط تُعيد النتيجة |

### ج) قاعدة الكتابة المركزية (Single-Writer Rule)

لتفادي تشتت الوصول لقاعدة البيانات عبر 8 Agents مختلفة (ممارسة غير احترافية)، الالتزام هو:

- **Application فقط** يكتب: الوثائق، المستخدمين، الحالة الأولية.
- **Orchestration فقط** يكتب: حالة الـ Workflow، نتائج الـ Agents.
- **لا Agent يتصل بـ Storage مباشرة** — أبدًا.
- **RAG Layer** له مخزن بيانات خاص به (ChromaDB) منفصل تمامًا عن Storage العلائقية — هذا مقصود، فبيانات المتجهات (Embeddings) مختلفة طبيعيًا عن البيانات العلائقية (Documents/Users/Results).

هذا التبسيط هو ما يجعل الربط "صحيحًا واحترافيًا وبسيطًا" في آن واحد: مسار تنفيذي واضح + خدمات مشتركة معزولة + نقطة كتابة واحدة لكل نوع بيانات.

**→ راجع `layer_connections_diagram.svg`** للتمثيل البصري الكامل لهذا الربط.

---

## قابلية التوسّع (Extensibility)

| السيناريو | التعديل المطلوب فقط |
|---|---|
| إضافة Agent جديد | مجلد جديد تحت `Agents/` يطبّق `BaseAgent` + تسجيله في `agent_registry.py` |
| استبدال Gemma 3 بنموذج آخر | داخل `Inference/models/` و `Inference/client/` فقط |
| تغيير Vector Database | داخل `RAG/vector_store/` فقط (تطبيق جديد للـ Interface) |
| إضافة قاعدة توجيه جديدة | داخل `Orchestration/supervisor/routing_logic.py` فقط |
| دعم قناة رفع جديدة (مثال: بريد إلكتروني) | إضافة Controller/Route جديد في Application يستدعي نفس `DocumentProcessingService` |

---

## Tests / Docker / Documentation / Config

```
Tests/
├── Presentation/     # Jest / Playwright
├── Application/      # اختبارات Unit/Integration لـ C++ Controllers/Services
├── Orchestration/     # اختبارات Workflow والـ Supervisor
├── Agents/            # اختبار كل Agent بمعزل عن غيره
├── RAG/                # دقة الاسترجاع (Retrieval Accuracy)
└── Integration/        # اختبارات End-to-End تعبر كل الطبقات

Docker/
├── Dockerfile.presentation
├── Dockerfile.application
├── Dockerfile.orchestration     # يشمل Orchestration + Agents (نفس Python runtime)
├── Dockerfile.inference          # بناء llama-server
├── Dockerfile.rag
└── docker-compose.override.yml   # بيئة تطوير محلي

Documentation/
├── architecture.md               # نسخة موسّعة من هذه الوثيقة
├── layer_connections_diagram.svg
├── agent_catalog.md              # كتالوج تفصيلي لكل Agent (مدخلات/مخرجات)
├── api_reference.md              # توثيق REST Endpoints
└── ADRs/
    └── 0001-choose-langgraph-for-orchestration.md

Config/
├── global_config.yaml            # منافذ، عناوين خدمات داخلية مشتركة
├── logging_config.yaml
└── secrets.env.example
```

---

## الخلاصة

البنية أعلاه توفر: طبقات معمارية واضحة الحدود، Agents مستقلة قابلة للإضافة دون تعديل الجوهر، نقاط استبدال محددة لكل من النموذج اللغوي وقاعدة المتجهات، وربطًا مبسّطًا بين الطبقات يقوم على **مسار تنفيذي واحد + خدمات مشتركة معزولة + كتابة مركزية لقاعدة البيانات**.
