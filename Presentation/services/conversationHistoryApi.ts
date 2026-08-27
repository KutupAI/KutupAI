import { ChatMessageModel } from "../models/ChatMessage";
import { ChatModel } from "../models/Chat";
import { normalizePipelineState } from "../models/AnalysisResult";

type ConversationListItem = {
  chat_id: string;
  title: string;
  updated_at: string;
};

type ConversationTurn = {
  question: string;
  answer: string;
  pipeline_state?: unknown;
  created_at: string;
};

type ConversationDetail = {
  chat_id: string;
  updated_at: string;
  turns: ConversationTurn[];
};

const baseUrl = (): string => {
  const configured = import.meta.env.VITE_MEMORY_API_BASE_URL?.replace(/\/$/, "");
  return configured || "/memory";
};

const toTimestamp = (value: string): number => {
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Date.now() : parsed;
};

const summaryToChat = (item: ConversationListItem): ChatModel => ({
  id: item.chat_id,
  title: item.title || "Yeni Sohbet",
  messages: [],
  updatedAt: toTimestamp(item.updated_at),
});

const detailToChat = (detail: ConversationDetail): ChatModel => {
  const messages: ChatMessageModel[] = detail.turns.flatMap((turn, index) => {
    const createdAt = toTimestamp(turn.created_at);
    return [
      {
        id: `${detail.chat_id}-${index}-question`,
        role: "user" as const,
        content: turn.question,
        createdAt,
        status: "sent" as const,
      },
      {
        id: `${detail.chat_id}-${index}-answer`,
        role: "assistant" as const,
        content: turn.answer,
        createdAt: createdAt + 1,
        status: "sent" as const,
        pipelineState: normalizePipelineState(turn.pipeline_state),
      },
    ];
  });
  return {
    id: detail.chat_id,
    title: detail.turns[0]?.question || "Yeni Sohbet",
    messages,
    updatedAt: toTimestamp(detail.updated_at),
  };
};

/** Orchestration SQLite hafızası için Sidebar istemcisi. */
export const conversationHistoryApi = {
  async list(): Promise<ChatModel[]> {
    const response = await fetch(`${baseUrl()}/conversations`);
    if (!response.ok) throw new Error("Conversation history is unavailable");
    const body = (await response.json()) as { items?: ConversationListItem[] };
    return Array.isArray(body.items) ? body.items.map(summaryToChat) : [];
  },

  async get(chatId: string): Promise<ChatModel> {
    const response = await fetch(`${baseUrl()}/conversations/${encodeURIComponent(chatId)}`);
    if (!response.ok) throw new Error("Conversation could not be loaded");
    return detailToChat((await response.json()) as ConversationDetail);
  },

  async remove(chatId: string): Promise<void> {
    const response = await fetch(`${baseUrl()}/conversations/${encodeURIComponent(chatId)}`, {
      method: "DELETE",
    });
    if (!response.ok && response.status !== 404) {
      throw new Error("Conversation could not be deleted");
    }
  },
};
