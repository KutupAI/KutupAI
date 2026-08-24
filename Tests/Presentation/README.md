# Presentation UI Preview (Vite) → Application Layer

## Live (موصى به)

1. شغّل Orchestration على `:8000`
2. شغّل Application على `:8080`
3. ثم:

```bash
cd Tests/Presentation
npm install
npm run dev
```

الواجهة على `http://localhost:5173` — طلبات `/api/*` تُمرَّر عبر Vite proxy إلى Application.

العقد: `POST /api/Chat/SendMessage` → `{ Success, AdditionalData: { ChatId, State } }`  
`State` = `{ request, ocr, classification, extraction, validation, rag, summary, routing, writing }`

## Demo (بدون Backend)

```bash
cp .env.demo .env
npm run dev
```

أو: `VITE_CHAT_DEMO=true npm run dev`
