import { ApiResponse } from "../models/ApiResponse";
import { ChatMessageModel } from "../models/ChatMessage";
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
  Answer: string;
}

/**
 * يرسل سؤال المستخدم (+ ملف اختياري + معرف المحادثة) إلى Application Layer.
 * منطق الذكاء الاصطناعي بالكامل خارج React — هذه الدالة تمرر الطلب فقط
 * وتعيد النتيجة عبر callback بنفس نمط باقي المشروع.
 */
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

  http.post<SendMessageResponseData>(
    CHAT_CONFIG.endpoints.sendMessage,
    model,
    (res) => {
      if (res.Success && res.AdditionalData) {
        onSuccess(res.AdditionalData);
      } else {
        onError(res);
      }
    }
  );
};

export type { ChatMessageModel };
