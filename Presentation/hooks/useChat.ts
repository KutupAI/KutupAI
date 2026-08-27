import { useCallback, useState } from "react";
import { ChatMessageModel } from "../models/ChatMessage";
import { ChatModel, createEmptyChat } from "../models/Chat";
import { sendChatMessage } from "../services/chatService";
import { CHAT_CONFIG } from "../config/chatConfig";

interface PendingFile {
  name: string;
  type: string;
  base64: string;
  previewUrl?: string;
}

const generateId = (): string =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

export const useChat = () => {
  const [currentChat, setCurrentChat] = useState<ChatModel>(createEmptyChat());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startNewChat = useCallback(() => {
    setCurrentChat(createEmptyChat());
    setError(null);
    setIsLoading(false);
  }, []);

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
          ? {
              name: pendingFile.name,
              type: pendingFile.type,
              previewUrl: pendingFile.previewUrl,
            }
          : undefined,
      };

      setCurrentChat((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));
      setIsLoading(true);

      sendChatMessage(
        currentChat.id,
        trimmed,
        pendingFile
          ? { name: pendingFile.name, type: pendingFile.type, base64: pendingFile.base64 }
          : null,
        (data) => {
          const answer =
            data.State.writing.answer?.trim()
            || data.State.summary.rag_summary_text?.trim()
            || "";
          const assistantMessage: ChatMessageModel = {
            id: generateId(),
            role: "assistant",
            content: answer,
            createdAt: Date.now(),
            status: "sent",
            pipelineState: data.State,
          };
          setCurrentChat((prev) => ({
            ...prev,
            id: data.ChatId ?? prev.id,
            messages: [...prev.messages, assistantMessage],
          }));
          setIsLoading(false);
        },
        (res) => {
          setError(res.Message || CHAT_CONFIG.ui.genericErrorLabel);
          setIsLoading(false);
        }
      );
    },
    [currentChat.id, isLoading]
  );

  return { currentChat, isLoading, error, startNewChat, sendMessage };
};
