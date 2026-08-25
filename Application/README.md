# Application Layer

**C++ · Drogon** — استقبال طلبات Presentation، التحقق، بناء عقد الحالة، استدعاء Orchestration، إرجاع النتيجة.  
**لا تحتوي أي منطق AI.**

القناة الحية: `POST /api/Chat/SendMessage` → المنفذ **8080** (يحتاج Orchestration على **8000**).

---

## كيف تعمل الطبقة (باختصار)

```text
Presentation
  → تحقق الشكل (JSON / ChatDTO)
  → تحقق العمل (سؤال / نوع / حجم الملف)
  → حفظ ملف مؤقت (Base64 → disk)
  → بناء عقد الحالة { request, ocr:{}, … writing:{} }
  → POST Orchestration /process
  → إرجاع { Success, AdditionalData: { ChatId, State } }
  → حذف الملف المؤقت دائمًا
```

| المدخل من Presentation | ما تفعله Application | المخرج إلى Presentation |
|---|---|---|
| `{ ChatId, Question, File? }` | تحقق + عقد + استدعاء Orchestration | `{ Success, AdditionalData: { ChatId, State } }` |

`File` = **ملف واحد فقط** (كائن وليس مصفوفة). لا دعم لرفع عدة صور دفعة واحدة في هذا العقد.

---

## المدخلات والمخرجات

### مدخل (من الواجهة)

```json
{
  "ChatId": null,
  "Question": "bu ne sozlesmesi",
  "File": {
    "FileName": "Elektrik sozlesmesi.pdf",
    "FileType": "application/pdf",
    "FileBase64": "<base64 بدون data: prefix>"
  }
}
```

- بدون ملف: أرسل `"File": null` و`Question` غير فارغ.
- مع ملف: `Question` يمكن أن يكون `""`.

### ما يُرسل إلى Orchestration (عقد الطبقة)

```json
{
  "request": {
    "success": true,
    "question": "bu ne sozlesmesi",
    "document": {
      "document_id": "req-...",
      "file_name": "Elektrik sozlesmesi.pdf",
      "file_type": "pdf"
    }
  },
  "ocr": {},
  "classification": {},
  "extraction": {},
  "validation": {},
  "rag": {},
  "summary": {},
  "routing": {},
  "writing": {}
}
```

(+ `document_path` داخليًا لمسار الملف المؤقت — لا يُعاد للواجهة.)

### مخرج (إلى الواجهة)

```json
{
  "Success": true,
  "AdditionalData": {
    "ChatId": "req-...",
    "State": {
      "request": {
        "success": true,
        "question": "bu ne sozlesmesi",
        "document": {
          "document_id": "req-...",
          "file_name": "Elektrik sozlesmesi.pdf",
          "file_type": "pdf"
        }
      },
      "ocr": {},
      "classification": {},
      "extraction": {},
      "validation": {},
      "rag": {},
      "summary": {},
      "routing": {},
      "writing": {}
    }
  }
}
```

عند الفشل: `{ "Success": false, "Message": "...", "Code": "APPLICATION_…" }` وغالبًا HTTP 400 (تحقق) أو 502 (Orchestration).

Orchestration داخليًا ما زال يستخدم `{ Success, Data }` — Application تحوّله إلى `AdditionalData.State` قبل الرد للواجهة.

---

## تجربة المدخلات والمخرجات

### 1) شغّل الطبقات

```powershell
# طرفية 1 — Orchestration
cd D:\AI\KutupAI
$env:ORCHESTRATION_PORT="8000"
python -m Orchestration.main

# طرفية 2 — Application
cd D:\AI\KutupAI\Application
$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="D:\AI\KutupAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"
.\build\Release\SmartGovernmentAI_Application.exe
```

### 2) حالات تجريبية (PowerShell)

**أ) سؤال فقط — يُرفض لاحقًا من Orchestration (لا مسار ملف):**
```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/Chat/SendMessage -Method POST -ContentType "application/json" `
  -Body '{"ChatId":null,"Question":"bu ne sozlesmesi","File":null}'
```

**ب) ملف صالح + سؤال (المسار السعيد):**
```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("D:\path\to\sample.pdf"))
$body = @{
  ChatId = $null
  Question = "bu ne sozlesmesi"
  File = @{ FileName = "sample.pdf"; FileType = "application/pdf"; FileBase64 = $b64 }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8080/api/Chat/SendMessage -Method POST -ContentType "application/json" -Body $body
```

**ج) ملف أكبر من الحد (افتراضي 10MB) → رفض في Application:**
```powershell
# أي ملف > 10MB
# الرسالة: File exceeds maximum allowed size of 10MB.
# HTTP 400
```

**د) نوع غير مدعوم (مثل `.exe` / `application/zip`) → رفض:**
```powershell
# الرسالة: Unsupported file type: ...
# HTTP 400
```

**هـ) عدة صور:** العقد يقبل `File` كائنًا واحدًا فقط. إرسال مصفوفة `File: [ ... ]` → خطأ شكل الطلب (400). لرفع أكثر من صورة يجب دمجها خارجيًا (PDF واحد) أو إرسال طلب منفصل لكل ملف.

غيّر الحد عبر البيئة: `$env:APP_MAX_UPLOAD_SIZE_MB="20"`

---

## الشروط التي تتحقق منها الطبقة

كل الشروط **قبل** استدعاء Orchestration.

### شكل الطلب
| الحالة | النتيجة |
|---|---|
| JSON غير صالح / `Question` ليست نصًا | 400 |
| `File` مصفوفة أو نوع خاطئ (ليس object/null) | 400 |
| `File` موجود لكن ينقص `FileName` / `FileType` / `FileBase64` | 400 |

### قواعد العمل
| الحالة | النتيجة |
|---|---|
| لا ملف + سؤال فارغ/مسافات فقط | رفض — السؤال مطلوب |
| لا سؤال ولا ملف | رفض |
| **أكثر من ملف / مصفوفة صور** | غير مدعوم — `File` واحد فقط |
| نوع MIME/امتداد غير مسموح | رفض (`Unsupported file type`) |
| `FileBase64` فارغ | رفض |
| الحجم بعد الفك التقديري **> 10MB** (أو `APP_MAX_UPLOAD_SIZE_MB`) | رفض (`File exceeds maximum…`) |
| مسار خبيث في `FileName` (`../`) | يُؤخذ اسم الملف فقط (تعقيم) |

**الأنواع المسموحة:** PDF, DOCX, PPTX, TXT, وصور (`image/*` أو امتدادات jpg/png/gif/webp/tiff/bmp/heic…).

### بعد التحقق
| الحالة | النتيجة |
|---|---|
| فشل فك Base64 / الكتابة على القرص | `APPLICATION_TEMP_STORAGE_FAILED` |
| Orchestration لا يرد / مهلة 300s | `APPLICATION_ORCHESTRATION_UNREACHABLE` → 502 |
| رد ليس `{ Success, Data[] }` | `ORCHESTRATION_ERROR` |

---

## التشغيل والبناء

```powershell
cd D:\AI\KutupAI\Application
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"
cmake --build build --config Release
```

| متغير | افتراضي |
|---|---|
| `APP_SERVER_PORT` | `8080` |
| `ORCHESTRATION_BASE_URL` | `http://127.0.0.1:8000` |
| `APP_TEMP_UPLOAD_ROOT_DIR` | مسار كامل لـ `Storage/files/temp_processing` |
| `ORCHESTRATION_TIMEOUT_SECONDS` | `300` |
| `APP_MAX_UPLOAD_SIZE_MB` | `10` |

لا تضع مسارًا فيه `...` — استخدم المسار الكامل.

---

## الملفات الأساسية

| الملف | الدور |
|---|---|
| `Controllers/ChatController` | استقبال الطلب وإرجاع `{ Success, Data }` |
| `Validators/DocumentValidator` | نوع/حجم/سؤال |
| `DTOs/LayerStateDTO` | عقد `{ request, ocr, … writing }` |
| `DTOs/ChatDTO` | شكل مدخل الواجهة |
| `Services/ChatService` | تدفق كامل: تحقق → temp → عقد → Orchestration → تنظيف |
| `Services/OrchestrationClient` | `POST /process` فقط |
| `Configuration/AppConfig` | المنافذ، المهلة، حد الحجم |

باقي Controllers/Routes (Document/Auth/User/Status) هياكل غير مربوطة في البناء بعد.
