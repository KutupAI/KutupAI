/**
 * chatHistoryStorage.ts
 * ----------------------
 * Sohbet geçmişi için depolama katmanı -- şu an `localStorage` kullanır
 * (Application layer'da henüz bir "sohbet listesi" endpoint'i yok), ama
 * arayüz (`ChatHistoryStorage`) kasıtlı olarak taşıma-bağımsız (transport-
 * agnostic) tasarlandı: ileride gerçek bir backend endpoint'i eklenirse,
 * SADECE bu dosyadaki `localStorageAdapter`'ın yerine bir
 * `serverAdapter` (fetch tabanlı, aynı arayüzü uygulayan) geçirilir --
 * `useChatHistory` hook'u veya herhangi bir UI bileşeni DEĞİŞMEZ.
 *
 * Sağlamlık (professional/defensive tasarım):
 * - Her localStorage erişimi try/catch içinde -- kota dolu, gizli
 *   gezinti (private browsing) localStorage'ı engelliyor, veya bozuk JSON
 *   gibi durumlarda ASLA uygulamayı çökertmez; sessizce boş listeye düşer.
 * - Geçmiş `MAX_HISTORY_ITEMS` ile sınırlanır -- sınırsız büyüyüp kota
 *   hatasına yol açmaz (en eski sohbetler otomatik düşer).
 * - Yazma işlemleri "son yazan kazanır" (last-write-wins) -- aynı sekmede
 *   çoklu hızlı güncellemede tutarsızlık olmaz çünkü her zaman güncel
 *   `chats` dizisinin tamamı yazılır, parça parça değil.
 */
import { ChatModel } from "../models/Chat";

const STORAGE_KEY = "kutup_chat_history_v1";
const MAX_HISTORY_ITEMS = 50;

export interface ChatHistoryStorage {
  load: () => ChatModel[];
  saveAll: (chats: ChatModel[]) => void;
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Depodan gelen ham veriyi doğrular -- şekli bozuksa o kaydı sessizce
 *  atlar (tek bir bozuk kayıt tüm geçmişi kaybettirmesin diye). */
const isValidChat = (v: unknown): v is ChatModel =>
  isRecord(v) &&
  typeof v.title === "string" &&
  Array.isArray(v.messages) &&
  typeof v.updatedAt === "number";

const localStorageAdapter: ChatHistoryStorage = {
  load: () => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(isValidChat);
    } catch {
      // Kota/izin/bozuk JSON hatası -- geçmiş olmadan devam et, çökme yok.
      return [];
    }
  },

  saveAll: (chats) => {
    try {
      const capped = [...chats]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .slice(0, MAX_HISTORY_ITEMS);
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(capped));
    } catch {
      // Yazma başarısız olabilir (kota dolu, gizli gezinti vb.) --
      // sohbet mevcut oturumda çalışmaya devam eder, sadece kalıcı olmaz.
    }
  },
};

export const chatHistoryStorage: ChatHistoryStorage = localStorageAdapter;
