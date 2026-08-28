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

## Güncel kullanıcı akışı

- Sohbet geçmişi Orchestration'ın kalıcı konuşma API'sinden alınır: `GET /conversations` ve `GET /conversations/{chat_id}`.
- Dosyasız ilk soru doğrudan RAG → Summary → Writing akışına gider.
- Dosya soru olmadan yüklenirse belge işlenir ve belge özeti üretilebilir.
- Her kullanıcı nihai yanıtı ve `Kaynak` kanıtlarını açabilir. Kanıt metni ilk anda kapalıdır.
- `Özet` alanı belge bilgileri, çıkarılan alanlar ve işlem adımlarını gösterir; yalnız yönetici görünümünde açılır.

## Demo yönetici görünümü

Bu görünüm yarışma demosu için yalnız istemci tarafında çalışan sabit bir kontroldür; gerçek bir kimlik doğrulama sistemi değildir. Oturum `sessionStorage` içinde tutulur ve sekme kapanınca silinir.

```text
E-posta: kutup@kutupai.local
Şifre:   159753
```

Üretim kullanımında bu kontrol sunucu tarafı kimlik doğrulama ve rol yetkilendirmesiyle değiştirilmelidir.

Live stack: Orchestration `:8000` → Application `:8080` → this UI. Endpoint: `POST /api/Chat/SendMessage`.
