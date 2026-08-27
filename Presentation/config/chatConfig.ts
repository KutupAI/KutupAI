/**
 * إعدادات KutupAI Chat — مربوطة بـ Application `POST /api/Chat/SendMessage`.
 */
export const CHAT_CONFIG = {
  endpoints: {
    sendMessage: "/api/Chat/SendMessage",
  },

  upload: {
    // أنواع الملفات التي تظهر في نافذة اختيار الملف
    acceptAttribute: ".pdf,.jpg,.jpeg,.png,.txt,.docx",

    // الامتدادات المسموح بها
    allowedExtensions: [
      "pdf",
      "jpg",
      "jpeg",
      "png",
      "txt",
      "docx",
    ],

    // MIME Types
    allowedMimeTypes: [
      "application/pdf",
      "image/jpeg",
      "image/png",
      "text/plain",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],

    maxSizeBytes: 10 * 1024 * 1024, // 10MB
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