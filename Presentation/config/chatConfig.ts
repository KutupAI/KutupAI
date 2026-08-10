/**
 * إعدادات KutupAI Chat.
 * عناوين الـ API أدناه Placeholders — استبدلها بالـ Endpoints الفعلية
 * الموجودة في Application Layer (DocumentController / ChatController...الخ)
 * دون تغيير http helper الحالي أو نمط الاستجابة.
 */
export const CHAT_CONFIG = {
  endpoints: {
    // TODO: استبدل بالـ endpoint الفعلي، مثال: '/Chat/SendMessage'
    sendMessage: "/api/Chat/SendMessage",
    // TODO: استبدل بالـ endpoint الفعلي لإنشاء محادثة جديدة إن وُجد على السيرفر
    createChat: "/api/Chat/CreateChat",
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
    maxSizeBytes: 10 * 1024 * 1024, // 10MB
  },
  ui: {
    productName: "KutupAI",
    tagline: "Türkiye için akıllı yapay zeka",
    thinkingLabel: "KutupAI düşünüyor...",
    genericErrorLabel: "Bir hata oluştu",
  },
} as const;
