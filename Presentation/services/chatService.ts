import { ApiResponse } from "../models/ApiResponse";
import {
  normalizePipelineState,
  PipelineState,
} from "../models/AnalysisResult";
import { CHAT_CONFIG } from "../config/chatConfig";
import { http } from "./http";

export interface SendMessageRequestModel {
  ChatId: string | null;
  Question: string;
  File?: {
    FileName: string;
    FileType: string;
    FileBase64: string;
  } | null;
}

export interface SendMessageResponseData {
  ChatId: string;
  /** Filled unified pipeline envelope from Application AdditionalData.State */
  State: PipelineState;
}

type WireAdditional = {
  ChatId?: string;
  State?: unknown;
  Data?: unknown[];
};


export const sendChatMessage = (
  chatId: string | null,
  question: string,
  file: { name: string; type: string; base64: string } | null,
  onSuccess: (data: SendMessageResponseData) => void,
  onError: (res: ApiResponse<SendMessageResponseData>) => void
): void => {
  const model: SendMessageRequestModel = {
    ChatId: chatId,
    Question: question,
    File: file
      ? { FileName: file.name, FileType: file.type, FileBase64: file.base64 }
      : null,
  };

  http.post<WireAdditional>(
    CHAT_CONFIG.endpoints.sendMessage,
    model,
    (res) => {
      if (!res.Success) {
        onError(res as ApiResponse<SendMessageResponseData>);
        return;
      }

      const raw = res.AdditionalData ?? {};
      let stateRaw: unknown = raw.State;

      // Fallback: legacy { Data: [doc] } nested or top-level
      if (stateRaw == null && Array.isArray(raw.Data) && raw.Data[0]) {
        stateRaw = raw.Data[0];
      }

      const state = normalizePipelineState(stateRaw);
      const resolvedChatId =
        typeof raw.ChatId === "string" && raw.ChatId
          ? raw.ChatId
          : chatId ||
            state.request.document?.document_id ||
            `chat-${Date.now()}`;

      onSuccess({ ChatId: resolvedChatId, State: state });
    }
  );
};
