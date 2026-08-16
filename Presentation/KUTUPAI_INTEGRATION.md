# KutupAI Chat — دليل التكامل السريع

هذه الملفات مبنية لتتوافق مع بنية `Presentation/` الموصوفة في وثيقة العمارة
(`components/`, `pages/`, `services/`, `hooks/`, `models/`, `utils/`, `config/`),
وتُضاف إلى المشروع الحالي دون كسر أي شيء موجود.

## 1) نسخ الملفات
انسخ محتوى هذا المجلد داخل `Presentation/` الحالي — الأسماء لا تتعارض مع
أي كومبوننت مذكور في الوثيقة (`DocumentUploader`, `AnalysisResultCard`...).

## 2) ربط الـ http helper الحقيقي
افتح:
```
services/chatService.ts
```
واستبدل السطرين:
```ts
declare const http: { post: <TData = unknown>(url: string, model: unknown, onResult: (res: ApiResponse<TData>) => void) => void };
```
بالاستيراد الفعلي من مكان الـ helper في مشروعك، مثال:
```ts
import { http } from "../../shared/http";
```

## 3) ربط Endpoint الحقيقي
في `config/chatConfig.ts` غيّر:
```ts
sendMessage: "/api/Chat/SendMessage"
```
إلى الـ Endpoint الفعلي في `Application/Controllers/DocumentController` أو
Controller مخصص للمحادثة إن وُجد.

## 4) تحميل الـ Theme
أضف استيراد واحد في نقطة الدخول الحالية (`main.tsx` أو `App.tsx`):
```ts
import "./styles/theme.css"; // أو المسار المطابق بعد النسخ
```

## 5) استخدام الصفحة
```tsx
import ChatPage from "./pages/ChatPage";
// اربطها بالـ Route المناسب في الـ Router الحالي، مثال:
// <Route path="/chat" element={<ChatPage />} />
```

## ملاحظات معمارية
- كل استدعاء API يمر حصراً عبر `services/chatService.ts` (لا استدعاء API
  مباشر من أي Component)، اتساقاً مع قاعدة `services/` في Presentation Layer.
- لا يوجد أي منطق AI داخل React — فقط UI/State/File Selection/API
  Request/Display، تماشياً مع "قاعدة صلبة" في Application Layer بالوثيقة.
- الـ Bootstrap مستخدم فقط للـ utility classes الأساسية (`btn`, layout)؛
  كل التصميم البصري (الألوان/الحواف/الحركة) عبر CSS Modules خاصة بكل
  كومبوننت — نفس نمط `DocumentUploader.module.css` الموجود في الوثيقة.

|
cd Tests/Presentation
npm install
npm run dev