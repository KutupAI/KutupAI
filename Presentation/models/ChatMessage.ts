import { PipelineState } from "./AnalysisResult";

export type ChatRole = "user" | "assistant";

export type MessageStatus = "sending" | "sent" | "error";

export interface ChatFileAttachment {
  name: string;
  type: string;
  sizeBytes: number;
  /** Base64 (بدون data: prefix) — يُستخدم لإرساله ضمن body الطلب JSON. */
  base64: string;
  /** Object URL محلي لعرض Thumbnail للصور فقط، لا يُرسل للـ API. */
  previewUrl?: string;
}

export interface ChatMessageModel {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  file?: Pick<ChatFileAttachment, "name" | "type" | "previewUrl">;
  status: MessageStatus;
  /** Filled 9-key pipeline envelope (assistant only). When present, UI shows
   *  Structured Response Format via AnalysisResultCard — never raw JSON. */
  pipelineState?: PipelineState;
}
