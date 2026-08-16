/**
 * HTTP bridge used by chatService.
 * In Tests/Presentation preview, VITE_CHAT_DEMO=true returns a local fake answer.
 */
import { ApiResponse } from "../models/ApiResponse";

type ResultCb<T> = (res: ApiResponse<T>) => void;

const isDemo =
  typeof import.meta !== "undefined" &&
  Boolean((import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_CHAT_DEMO);

export const http = {
  post<TData = unknown>(
    url: string,
    model: unknown,
    onResult: ResultCb<TData>
  ): void {
    if (isDemo) {
      const body = model as { Question?: string; ChatId?: string | null };
      window.setTimeout(() => {
        onResult({
          Success: true,
          Message: "demo",
          AdditionalData: {
            ChatId: body.ChatId || `demo-${Date.now()}`,
            Answer:
              `Bu bir önizleme yanıtıdır (Tests/Presentation).\n\n` +
              `Sorunuz: "${body.Question ?? ""}"\n\n` +
              `- Gerçek AI / Application API henüz bağlı değil\n` +
              `- Arayüzü buradan kontrol edebilirsiniz`,
          } as TData,
        });
      }, 700);
      return;
    }

    // Real backend path (when demo is off)
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(model),
    })
      .then(async (r) => {
        const data = (await r.json()) as ApiResponse<TData>;
        onResult(data);
      })
      .catch((err: Error) => {
        onResult({
          Success: false,
          Message: err.message || "Network error",
        });
      });
  },
};
