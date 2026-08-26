# Presentation Layer

TypeScript / React / Vite. `services/` is the only place that calls the Application API.

## Stack (used in code)

| Package | Role |
|---|---|
| `react` / `react-dom` | UI |
| `vite` + `@vitejs/plugin-react` | bundler / HMR |
| `typescript` | types |

No router, state library, or HTTP client packages — chat uses React hooks + native `fetch`.

## Run

```bash
cd Presentation
npm install
npm run dev          # http://localhost:5173 — proxies /api → :8080
npm run build        # typecheck + production bundle
```

Demo without backend: `VITE_CHAT_DEMO=true npm run dev`

Live stack: Orchestration `:8000` → Application `:8080` → this UI. Endpoint: `POST /api/Chat/SendMessage`.
