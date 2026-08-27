/**
 * إعدادات KutupAI Chat — مربوطة بـ Application `POST /api/Chat/SendMessage`.
 */
export const CHAT_CONFIG = {
  endpoints: {
    sendMessage: "/api/Chat/SendMessage",
  },
  upload: {
    acceptAttribute: ".pdf,.jpg,.jpeg,.png,.txt",
    allowedExtensions: ["pdf", "jpg", "jpeg", "png", "txt"],
    allowedMimeTypes: [
      "application/pdf",
      "image/jpeg",
      "image/png",
      "text/plain",
    ],
    maxSizeBytes: 10 * 1024 * 1024, // 10MB — mirrors Application AppConfig
  },
  ui: {
    productName: "KutupAI",
    tagline: "Türkiye için akıllı yapay zeka",
    thinkingLabel: "KutupAI düşünüyor...",
    genericErrorLabel: "Bir hata oluştu",
    /** Shown in order while waiting (no live progress from backend). */
    thinkingStages: [
      "Belge okunuyor...",
      "İçerik çıkarılıyor...",
      "Sınıflandırılıyor...",
      "Mevzuat aranıyor...",
      "Özet hazırlanıyor...",
      "Yanıt yazılıyor...",
    ],
  },
} as const;
