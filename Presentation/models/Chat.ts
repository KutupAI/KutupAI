import { ChatMessageModel } from "./ChatMessage";

export interface ChatModel {
  id: string | null;
  title: string;
  messages: ChatMessageModel[];
  /** Epoch ms — son mesajın (veya oluşturulma anının) zamanı. Sidebar'da
   *  gruplama (Bugün / Dün / Bu hafta / Daha eski) ve sıralama için. */
  updatedAt: number;
}

export const createEmptyChat = (): ChatModel => ({
  id: null,
  title: "Yeni Sohbet",
  messages: [],
  updatedAt: Date.now(),
});

/** İlk kullanıcı mesajından kısa, okunabilir bir sohbet başlığı üretir --
 *  Claude/ChatGPT tarzı: tam cümle değil, 40 karaktere kırpılmış özet. */
export const deriveChatTitle = (firstUserMessage: string): string => {
  const cleaned = firstUserMessage.trim().replace(/\s+/g, " ");
  if (!cleaned) return "Yeni Sohbet";
  return cleaned.length > 42 ? `${cleaned.slice(0, 42).trimEnd()}…` : cleaned;
};
