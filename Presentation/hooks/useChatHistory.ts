import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatMessageModel } from "../models/ChatMessage";
import { ChatModel, createEmptyChat, deriveChatTitle } from "../models/Chat";
import { sendChatMessage } from "../services/chatService";
import { chatHistoryStorage } from "../services/chatHistoryStorage";
import { CHAT_CONFIG } from "../config/chatConfig";

interface PendingFile {
  name: string;
  type: string;
  base64: string;
  previewUrl?: string;
}

const generateId = (): string =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

/**
 * useChatHistory — useChat'in yerini alır, aynı davranışı korur
 * (currentChat / isLoading / error / startNewChat / sendMessage) ve
 * üstüne sohbet geçmişi yönetimini ekler: `chats` (sidebar listesi),
 * `activeChatId`, `selectChat`, `deleteChat`.
 *
 * Tasarım kararı: boş ("Yeni Sohbet", hiç mesajı olmayan) bir sohbet
 * `chats` listesine EKLENMEZ -- kullanıcı "Yeni Sohbet"e her bastığında
 * sidebar'da boş satırlar birikmesin diye. Bir sohbet, ilk mesaj
 * gönderildiği an listeye girer (başlığı o mesajdan türetilerek).
 */
export const useChatHistory = () => {
  const [chats, setChats] = useState<ChatModel[]>(() => chatHistoryStorage.load());
  const [draftChat, setDraftChat] = useState<ChatModel>(() => createEmptyChat());
  const [activeChatId, setActiveChatId] = useState<string | null>(null); // null = draft (unsaved)
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // İlk yüklemeden sonraki her `chats` değişiminde kalıcı depoya yaz.
  const hydratedRef = useRef(false);
  useEffect(() => {
    if (!hydratedRef.current) {
      hydratedRef.current = true;
      return;
    }
    chatHistoryStorage.saveAll(chats);
  }, [chats]);

  const currentChat: ChatModel = useMemo(() => {
    if (!activeChatId) return draftChat;
    return chats.find((c) => c.id === activeChatId) ?? draftChat;
  }, [activeChatId, chats, draftChat]);

  const startNewChat = useCallback(() => {
    setDraftChat(createEmptyChat());
    setActiveChatId(null);
    setError(null);
    setIsLoading(false);
  }, []);

  const selectChat = useCallback(
    (id: string) => {
      if (isLoading) return; // Yanıt beklerken sohbet değiştirmeyi engelle (yarım kalmış istek karışmasın).
      setActiveChatId(id);
      setError(null);
    },
    [isLoading]
  );

  const deleteChat = useCallback(
    (id: string) => {
      setChats((prev) => prev.filter((c) => c.id !== id));
      if (activeChatId === id) {
        setDraftChat(createEmptyChat());
        setActiveChatId(null);
      }
    },
    [activeChatId]
  );

  const sendMessage = useCallback(
    (text: string, pendingFile: PendingFile | null) => {
      const trimmed = text.trim();
      if (!trimmed || isLoading) return;

      setError(null);

      const userMessage: ChatMessageModel = {
        id: generateId(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
        status: "sent",
        file: pendingFile
          ? { name: pendingFile.name, type: pendingFile.type, previewUrl: pendingFile.previewUrl }
          : undefined,
      };

      const isFirstMessage = currentChat.messages.length === 0;
      const localId = currentChat.id ?? `local-${generateId()}`;

      const chatAfterUserMsg: ChatModel = {
        ...currentChat,
        id: localId,
        title: isFirstMessage ? deriveChatTitle(trimmed) : currentChat.title,
        messages: [...currentChat.messages, userMessage],
        updatedAt: Date.now(),
      };

      // İlk mesajdan itibaren sohbet artık "kalıcı" -- listeye gir/güncelle.
      setChats((prev) => {
        const exists = prev.some((c) => c.id === localId);
        return exists
          ? prev.map((c) => (c.id === localId ? chatAfterUserMsg : c))
          : [chatAfterUserMsg, ...prev];
      });
      setActiveChatId(localId);
      setIsLoading(true);

      sendChatMessage(
        currentChat.id,
        trimmed,
        pendingFile
          ? { name: pendingFile.name, type: pendingFile.type, base64: pendingFile.base64 }
          : null,
        (data) => {
          const answer =
            data.State.writing.answer?.trim() || data.State.summary.rag_summary_text?.trim() || "";
          const assistantMessage: ChatMessageModel = {
            id: generateId(),
            role: "assistant",
            content: answer,
            createdAt: Date.now(),
            status: "sent",
            pipelineState: data.State,
          };
          const finalId = data.ChatId ?? localId;

          setChats((prev) => {
            const withoutOld = prev.filter((c) => c.id !== localId && c.id !== finalId);
            const updated: ChatModel = {
              ...chatAfterUserMsg,
              id: finalId,
              messages: [...chatAfterUserMsg.messages, assistantMessage],
              updatedAt: Date.now(),
            };
            return [updated, ...withoutOld];
          });
          setActiveChatId(finalId);
          setIsLoading(false);
        },
        (res) => {
          setError(res.Message || CHAT_CONFIG.ui.genericErrorLabel);
          setIsLoading(false);
        }
      );
    },
    [currentChat, isLoading]
  );

  return {
    currentChat,
    chats,
    activeChatId,
    isLoading,
    error,
    startNewChat,
    selectChat,
    deleteChat,
    sendMessage,
  };
};