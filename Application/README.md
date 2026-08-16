# Application Layer

**اللغة:** C++ · **الإطار:** Drogon  
**المسؤولية:** استقبال طلبات Presentation عبر REST، التحقق من الملف/السؤال، حفظ الملف مؤقتًا، استدعاء Orchestration، وإرجاع النتيجة الجاهزة.

هذه الطبقة **لا تحتوي أي منطق AI**. لا تصنيف، لا OCR، لا RAG.

القناة الحية الحالية: `POST /api/Chat/SendMessage`.  
القنوات Document/Auth/User/Status موجودة كهيكل ولم تُربَط في `CMakeLists.txt` بعد.

---

## التدفق

```text
Presentation  POST /api/Chat/SendMessage
      ↓
Routes/ChatRoutes.cpp
      ↓
Controllers/ChatController.cpp     تحليل JSON هيكلي
      ↓
Validators/DocumentValidator.cpp   نوع الملف / الحجم / السؤال
      ↓
Services/ChatService.cpp           فك Base64 + ملف مؤقت
      ↓
Services/OrchestrationClient.cpp   POST http://127.0.0.1:8000/process
      ↓
إرجاع { Success, Data } كما وصل من Orchestration
      ↓
حذف المجلد المؤقت
```

العقد المُرجَع إلى Presentation هو نفسه عقد OCR/Orchestration:

```json
{ "Success": true, "Data": [ { "document_id": "", "full_text": "", "pages": [], "...": "" } ] }
```

`Success = true` إذا اكتملت المعالجة. `false` إذا فشل الطلب (مسار ناقص، مهلة، ملف غير صالح).

---

## تشغيل الطبقة

المتطلبات في `requirements.txt`. بعد تثبيت vcpkg + Drogon:

```powershell
cd Application
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release

$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"
.\build\Release\SmartGovernmentAI_Application.exe
```

المنفذ **8080**. يحتاج Orchestration على **8000**.

لا تستخدم مسارًا فيه `C:\...` — ضع المسار الكامل لمجلد `Storage\files\temp_processing`.

---

## وظيفة كل ملف

### الجذر

| الملف | الوظيفة |
|---|---|
| `main.cpp` | تشغيل خادم Drogon: تحميل الإعدادات، بناء ChatService/Controller، تسجيل المسارات، الاستماع على المنفذ. |
| `CMakeLists.txt` | بناء التنفيذي وربط Drogon. الملفات المعلّقة هي الهياكل غير المربوطة بعد. |
| `requirements.txt` | متطلبات C++ / vcpkg / متغيرات البيئة. |
| `README.md` | هذا الملف. |

### `Controllers/`

| الملف | الوظيفة |
|---|---|
| `ChatController.h/.cpp` | استقبال `SendMessage`، تحويل الجسم إلى DTO، استدعاء ChatService، إرجاع `{ Success, Data }`. |
| `DocumentController.h/.cpp` | هيكل رفع/استعلام وثيقة عبر Storage (غير مربوط حاليًا). |
| `AuthController.h/.cpp` | هيكل تسجيل الدخول وإصدار التوكن. |
| `UserController.h/.cpp` | هيكل عمليات المستخدم. |
| `StatusController.h/.cpp` | هيكل قراءة حالة معالجة وثيقة من Storage. |

### `Routes/`

| الملف | الوظيفة |
|---|---|
| `ChatRoutes.h/.cpp` | ربط `POST /api/Chat/SendMessage` بـ `ChatController::sendMessage`. |
| `ApiRoutes.h` | تجميع تسجيل كل الـ Routes. |
| `DocumentRoutes.cpp` | هيكل مسارات الوثائق. |
| `AuthRoutes.cpp` | هيكل مسارات المصادقة. |

### `DTOs/`

| الملف | الوظيفة |
|---|---|
| `ChatDTO.h/.cpp` | شكل الطلب من الواجهة: `ChatId`, `Question`, `File { FileName, FileType, FileBase64 }`. الاستجابة تمرّر عقد `{ Success, Data }`. |
| `ApiResponseDTO.h/.cpp` | غلاف العقد: `documentEnvelope` / `emptyDocumentEnvelope`. دوال `success`/`failure` القديمة للقنوات الأخرى. |
| `DocumentRequestDTO.h` · `DocumentResponseDTO.h` | هيكل DTO رفع الوثيقة. |
| `AuthDTO.h` | هيكل DTO المصادقة. |
| `UserDTO.h` | هيكل DTO المستخدم. |

### `Services/`

| الملف | الوظيفة |
|---|---|
| `ChatService.h/.cpp` | مسار الدردشة: تحقق → حفظ مؤقت → Orchestration → تنظيف. الجسر الوحيد لهذه القناة نحو Orchestration. |
| `OrchestrationClient.h/.cpp` | العميل الوحيد الذي يعرف `POST /process`. يمرّر `{ document_id, question, document_path }` وينتظر `{ Success, Data }`. |
| `DocumentProcessingService.h/.cpp` | هيكل جسر قناة الرفع المعتمدة على Storage. |
| `AuthService.h/.cpp` | هيكل منطق الهوية والجلسات. |
| `UserService.h/.cpp` | هيكل منطق المستخدم. |

### `Validators/`

| الملف | الوظيفة |
|---|---|
| `DocumentValidator.h/.cpp` | نوع MIME/الامتداد، حجم الملف، أن السؤال غير فارغ إذا لم يُرفق ملف. |
| `RequestValidator.h/.cpp` | تحقق هيكلي عام للطلبات. |

### `Middleware/`

| الملف | الوظيفة |
|---|---|
| `AuthMiddleware.h/.cpp` | هيكل التحقق من التوكن قبل الـ Controllers. |
| `LoggingMiddleware.h/.cpp` | هيكل تسجيل الطلب/الاستجابة. |
| `ErrorHandlingMiddleware.h/.cpp` | هيكل توحيد أخطاء HTTP. |
| `CorsMiddleware.h/.cpp` | هيكل CORS. |

### `Configuration/`

| الملف | الوظيفة |
|---|---|
| `AppConfig.h/.cpp` | المنفذ، عنوان Orchestration، مهلة الانتظار (300 ثانية)، مجلد الرفع المؤقت، الحد الأقصى لحجم الملف. يتجاهل المسارات الناقصة التي تحتوي `...`. |
| `DatabaseConfig.h` | هيكل إعدادات قاعدة البيانات. |
