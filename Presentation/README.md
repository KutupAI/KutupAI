# Presentation Layer

TypeScript / React. `services/` is the only place that calls the Application API.

## Live against Application

```powershell
# Terminal 1 — Orchestration :8000
cd D:\AI\KutupAI
$env:ORCHESTRATION_PORT="8000"
python -m Orchestration.main

# Terminal 2 — Application :8080
cd D:\AI\KutupAI\Application
$env:ORCHESTRATION_BASE_URL="http://127.0.0.1:8000"
$env:APP_TEMP_UPLOAD_ROOT_DIR="D:\AI\KutupAI\Storage\files\temp_processing"
$env:APP_SERVER_PORT="8080"
.\build\Release\SmartGovernmentAI_Application.exe

# Terminal 3 — UI
cd D:\AI\KutupAI\Tests\Presentation
npm install
npm run dev
```

Vite proxies `/api` → `http://127.0.0.1:8080`. Endpoint: `POST /api/Chat/SendMessage`.

Demo without backend: set `VITE_CHAT_DEMO=true` (see `Tests/Presentation/.env.demo`).
