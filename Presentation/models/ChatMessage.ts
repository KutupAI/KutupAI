import { PipelineState } from "./AnalysisResult";

export type ChatRole = "user" | "assistant";

export type MessageStatus = "sending" | "sent" | "error";

export interface ChatFileAttachment {
  name: string;
  type: string;
  sizeBytes: number;
  base64: string;
  previewUrl?: string;
}

export interface ChatMessageModel {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  file?: Pick<ChatFileAttachment, "name" | "type" | "previewUrl">;
  status: MessageStatus;
  /** Filled 9-key pipeline envelope (assistant only). "Detay" (özet metni)
   *  ve "Kaynak" (alıntılar) buradan, YEREL olarak (yeni bir istek
   *  atmadan) açılır-kapanır kutularla gösterilir -- bkz. ChatMessage.tsx. */
  pipelineState?: PipelineState;
}