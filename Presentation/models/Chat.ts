import { ChatMessageModel } from "./ChatMessage";

export interface ChatModel {
  id: string | null;
  title: string;
  messages: ChatMessageModel[];
}

export const createEmptyChat = (): ChatModel => ({
  id: null,
  title: "Yeni Sohbet",
  messages: [],
});
