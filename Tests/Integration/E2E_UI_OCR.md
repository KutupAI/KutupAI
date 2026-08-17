# E2E Test — Presentation → Application → Orchestration → OCR Agent

اختبار كامل من الواجهة مع حفظ `UnifiedOCRResult` كـ JSON.

## ⚠️ تعارض المنافذ

| خدمة | المنفذ |
|---|---|
| **Application (Drogon)** | **8080** |
| Gemma llama-server (`Inference/llamastart.bat`) | 8080 أيضاً |
| Orchestration | 8000 |
| Qwen-VL OCR (`qwen_vl_launcher.bat`) | 8081 |
| Presentation (Vite) | 5173 |

لاختبار OCR من الواجهة **لا تحتاج Gemma على 8080**.  
أوقف `llamastart.bat` واترك **Application** يستخدم 8080.

---

## 1) Qwen-VL (اختياري — للوثائق غير الواضحة)

```powershell
Agents\ocr_agent\qwen_vl_launcher.bat
```

تحقق: `http://127.0.0.1:8081/v1/models`

---

## 2) Orchestration

```powershell
cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI
pip install -r Orchestration\requirements.txt
pip install -r Agents\ocr_agent\requirements.txt

$env:ORCHESTRATION_PORT="8000"
$env:QWEN_VL_ENABLED="true"
$env:QWEN_VL_ENDPOINT="http://127.0.0.1:8081/v1/chat/completions"

python -m Orchestration.main
```

تحقق: `http://127.0.0.1:8000/health`

---

## 3) Application (C++)

```powershell
cd Application
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"

.\build\Release\SmartGovernmentAI_Application.exe
```

(أو `.\build\SmartGovernmentAI_Application.exe` حسب مخرجات CMake)

---

## 4) Presentation

```powershell
cd Tests\Presentation
npm install
npm run dev
```

- `.env.local` يضبط `VITE_CHAT_DEMO=false`
- Vite proxy يوجّه `/api` → `http://127.0.0.1:8080`

افتح: `http://localhost:5173`

---

## 5) التجربة من الواجهة

1. ارفع **PDF أو صورة** (`.pdf`, `.jpg`, `.png`)
2. أضف **نصًا اختياريًا** أو اتركه فارغًا
3. اضغط **إرسال**
4. بعد النجاح:
   - يُحمَّل ملف **`*.ocr.json`** تلقائيًا إلى مجلد **Downloads**
   - النص المستخرج يظهر في المحادثة
   - افتح DevTools → Console لرؤية `[KutupAI] OcrResult JSON downloaded`

---

## 6) التدفق

```
Presentation (5173)
  POST /api/Chat/SendMessage
    ↓ proxy
Application (8080) — temp file + cleanup
  POST /process
    ↓
Orchestration (8000) — OCRAgent
    ↓
OCRClient → OCRProcessor
  Unstructured → Quality Check → (Qwen 8081 إن لزم)
    ↓
UnifiedOCRResult → OcrResult في الاستجابة → تحميل JSON
```

---

## 7) اختبارات بدون واجهة (تحقق سريع)

```powershell
python Tests\Orchestration\test_process.py
python Tests\Integration\test_upload_ocr_flow.py
```

---

## 8) استكشاف الأخطاء

| المشكلة | الحل |
|---|---|
| Demo response | تأكد `Tests/Presentation/.env.local` → `VITE_CHAT_DEMO=false` وأعد `npm run dev` |
| Network error | Application/Orchestration غير شغّالين |
| 502 Bad Gateway | Orchestration فشل — راجع terminal Orchestration |
| unstructured not installed | `pip install -r Agents\ocr_agent\requirements.txt` |
| Qwen لا يُستدعى | وثيقة واضحة = طبيعي (`meta.qwen_used: false`) |
| لا JSON | تأكد أنك رفعت **ملفًا** وليس نصًا فقط |


أكيد. بما إنك جهزت كل شيء، هاي طريقة التشغيل الكاملة من الصفر وبالترتيب، وكل خدمة خليها بــTerminal منفصل.

① Qwen-VL — المنفذ 8081

افتح Terminal جديد، ومن جذر المشروع:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI

شغّل:

.\Agents\ocr_agent\qwen_vl_launcher.bat

اترك النافذة مفتوحة.

للتأكد:

http://127.0.0.1:8081/v1/models

إذا ظهر JSON، فـQwen شغال ✅

لا تشغل Gemma على 8080 أثناء الاختبار، لأن Application يحتاج 8080.

② Orchestration — المنفذ 8000

افتح Terminal ثاني:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI

إذا أول مرة أو ناقص packages:

pip install -r Orchestration\requirements.txt
pip install -r Agents\ocr_agent\requirements.txt

ثم:

$env:ORCHESTRATION_PORT="8000"
$env:QWEN_VL_ENABLED="true"
$env:QWEN_VL_ENDPOINT="http://127.0.0.1:8081/v1/chat/completions"


python -m Orchestration.main

لا تسكر النافذة.

المفروض تشوف:

Uvicorn running on http://127.0.0.1:8000

ملاحظة: إذا /health أعطاك 404 مثل ما صار معك، هذا لا يعني أن السيرفر متوقف. المهم ظهور Uvicorn running....

③ Application — المنفذ 8080

افتح Terminal ثالث:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Application

أنت عملت الـBuild بنجاح، لذلك ما تحتاج تعيد CMake كل مرة.

شغّل فقط:

$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"


.\build\Release\SmartGovernmentAI_Application.exe

المفروض يظهر:

SmartGovernmentAI Application layer starting on port 8080
(Orchestration at http://127.0.0.1:8000)

خلي النافذة مفتوحة.

④ Presentation — المنفذ 5173

افتح Terminal رابع:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Tests\Presentation

تأكد أن عندك .env.local وفيه:

VITE_CHAT_DEMO=false

إذا ما عندك الملف، أنشئه:

Tests/
└── Presentation/
    ├── .env.local
    ├── package.json
    ├── vite.config.ts
    └── ...

ومحتواه:

VITE_CHAT_DEMO=false

ثم:

npm run dev

إذا أول مرة وما عملت npm install:

npm install
npm install -D @types/node
npm run dev

المفروض يعطيك:

Local: http://localhost:5173/

--------------------------
1. Python + مكتبات المشروع

ثبّت Python، وبعدها من جذر المشروع:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI


pip install -r Orchestration\requirements.txt
pip install -r Agents\ocr_agent\requirements.txt
2. Node.js + npm

لـPresentation:

cd Tests\Presentation


npm install

وبعدين:

npm run dev
3. CMake + Visual Studio

للـApplication المكتوب C++ تحتاج:

Visual Studio مع Desktop development with C++
CMake
Windows SDK

وبعدين تحتاج vcpkg لتثبيت Drogon ومكتباته.

أنت عندك عملتها هكذا:

C:\vcpkg\vcpkg.exe install drogon:x64-windows

وبناء Application:

cd C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Application


cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:\vcpkg\scripts\buildsystems\vcpkg.cmake"


cmake --build build --config Release
4. llama.cpp / Qwen-VL

وهذا أهم جزء للـOCR.

عندك مجلد:

Inference/
└── llama_server/
    ├── llama-server.exe
    ├── llama-server-impl.dll
    ├── llama.dll
    ├── ggml.dll
    ├── ggml-base.dll
    ├── ggml-cpu.dll
    ├── ...

يعني لا يكفي تنزل llama-server.exe لحاله؛ لازم ملفات DLL المطلوبة معه.

وعندك موديلات Qwen:

Agents/
└── ocr_agent/
    └── ocr_model/
        ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
        └── mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf

ثم qwen_vl_launcher.bat يشغل Qwen على:

127.0.0.1:8081
التشغيل على الجهاز الثاني

بعد تجهيز كل شيء، بتفتح 4 Terminals:

Terminal 1 — Qwen
cd C:\...\SmartGovernmentAI
.\Agents\ocr_agent\qwen_vl_launcher.bat

لازم يكون:

Qwen → 8081

Terminal 2 — Orchestration
cd C:\...\SmartGovernmentAI


$env:ORCHESTRATION_PORT="8000"
$env:QWEN_VL_ENABLED="true"
$env:QWEN_VL_ENDPOINT="http://127.0.0.1:8081/v1/chat/completions"


python -m Orchestration.main

يعني:

Orchestration → 8000

Terminal 3 — Application
cd C:\...\SmartGovernmentAI\Application


$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="C:\Users\SSCPrgWeb\Desktop\SmartGovernmentAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"


.\build\Release\SmartGovernmentAI_Application.exe

يعني:

Application → 8080
Terminal 4 — Presentation
cd C:\...\SmartGovernmentAI\Tests\Presentation


npm run dev

يعني:

Presentation → 5173

والـ.env.local:

VITE_CHAT_DEMO=false

والـhttp.ts لازم يكون مصحح بحيث:

const isDemo =
  typeof import.meta !== "undefined" &&
  (import.meta as ImportMeta & { env?: Record<string, string> })
    .env?.VITE_CHAT_DEMO === "true";
الأشياء التي لازم تنقلها للجهاز الثاني

إذا بدك تعمل Setup قابل للتكرار، أهم شيء لا تنسى:

SmartGovernmentAI/
│
├── Application/
│   └── CMakeLists.txt + C++ source
│
├── Orchestration/
│   ├── requirements.txt
│   └── Python source
│
├── Agents/
│   └── ocr_agent/
│       ├── requirements.txt
│       ├── ocr_model/
│       │   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
│       │   └── mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf
│       └── qwen_vl_launcher.bat
│
├── Inference/
│   └── llama_server/
│       ├── llama-server.exe
│       └── جميع DLLs المطلوبة
│
├── Presentation/
│
├── Tests/
│   └── Presentation/
│       └── .env.local
│
└── Storage/
    └── files/
        └── temp_processing/
الأدوات الخارجية التي لازم تكون على الجهاز الثاني
الأداة	الاستخدام
Python	Orchestration + OCR Agent
Node.js/npm	Presentation
Visual Studio C++	Application
CMake	Build Application
vcpkg	تثبيت Drogon
Drogon	Application HTTP server
llama.cpp	تشغيل Qwen-VL
Qwen-VL GGUF + mmproj GGUF	OCR/vision model

ملاحظة مهمة: لا تحتاج Gemma حتى تختبر مسار الـOCR الحالي؛ المسار الذي اشتغلنا عليه هو Qwen-VL على 8081. Gemma كان عندك على 8080 وتعارض مع Application، وهو أصلًا ليس جزءًا من مسار OCR الحالي.

إذا هدفك أن تعطي شخصًا آخر المشروع ويعمله بأقل تدخل ممكن، الأفضل بعدين نعمل SETUP.md + start_all.bat بحيث على الجهاز الثاني يثبت المتطلبات ثم يشغّل الطبقات الأربع تلقائيًا بدل فتح 4 نوافذ يدويًا.