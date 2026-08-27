/**
 * HTTP bridge used by chatService → Application Layer.
 *
 * - Live: POST to Application (via Vite `/api` proxy or VITE_APP_API_BASE_URL)
 * - Demo: VITE_CHAT_DEMO=true returns a local fake AdditionalData payload
 */
import { ApiResponse } from "../models/ApiResponse";
import { resolveApiUrl } from "../config/apiConfig";

type ResultCb<T> = (res: ApiResponse<T>) => void;

const isDemo =
  typeof import.meta !== "undefined" &&
  (import.meta as ImportMeta & { env?: Record<string, string> }).env?.VITE_CHAT_DEMO ===
    "true";

/** Application may return either AdditionalData or legacy { Success, Data }. */
type WireResponse<T> = ApiResponse<T> & {
  Data?: unknown[];
};

export const http = {
  post<TData = unknown>(
    url: string,
    model: unknown,
    onResult: ResultCb<TData>
  ): void {
    if (isDemo) {
      const body = model as {
        Question?: string;
        ChatId?: string | null;
        File?: { FileName: string; FileType: string } | null;
      };
      window.setTimeout(() => {
        const question = body.Question ?? "";
        const fileName = body.File?.FileName || "Elektrik sozlesmesi.pdf";
        const fileType =
          body.File?.FileType?.includes("/")
            ? body.File.FileType.split("/").pop() || "pdf"
            : body.File?.FileType || "pdf";
        const answer =
          `Bu bir önizleme yanıtıdır (Tests/Presentation demo).\n\n` +
          `Sorunuz: "${question}"`;

        onResult({
          Success: true,
          Message: "demo",
          AdditionalData: {
            ChatId: body.ChatId || `demo-${Date.now()}`,
            State: {
              request: {
                success: true,
                question,
                document: {
                  document_id: "DOC-001",
                  file_name: fileName,
                  file_type: fileType,
                },
              },
              ocr: {
                success: true,
                ocr_data: {
                  page_count: 1,
                  language: "tr",
                  pages: [],
                  full_text: "",
                  vision: {
                    signature: { detected: true, handwritten: true },
                    stamp: { detected: false },
                  },
                },
              },
              classification: {
                success: true,
                document_type: "Elektrik sozlesmesi",
                classification_confidence: 0.95,
              },
              extraction: {
                success: true,
                sender: null,
                date: null,
                address: null,
                phone: null,
                email: null,
              },
              validation: {
                success: true,
                is_complete: false,
                errors: [],
                warnings: [],
              },
              rag: {
                success: true,
                rag_data: {
                  operation: "retrieve",
                  query: question,
                  // Not: gerçek Application/RAG servisi henüz sonuç
                  // döndürmüyorsa "Kaynak" kutusu boş kalır (kart
                  // gizlenir) -- bu demo verisi sadece Presentation'ı
                  // yalıtılmış olarak test edebilmek için örnek içerik
                  // sağlar.
                  results: [
                    {
                      chunk_id: "demo-chunk-1",
                      law_number: "6098",
                      law_name: "Türk Borçlar Kanunu",
                      article_no: "1",
                      source_file: "6098_Turk Borclar Kanunu.pdf",
                      page_start: 12,
                      page_end: 12,
                      snippet:
                        "Sözleşme, tarafların iradelerini karşılıklı ve birbirine uygun olarak açıklamalarıyla kurulur.",
                      score: 0.91,
                    },
                    {
                      chunk_id: "demo-chunk-2",
                      law_number: "6098",
                      law_name: "Türk Borçlar Kanunu",
                      article_no: "27",
                      source_file: "6098_Turk Borclar Kanunu.pdf",
                      page_start: 15,
                      page_end: 16,
                      snippet: null,
                      score: 0.82,
                    },
                  ],
                },
              },
              summary: {
                success: true,
                rag_summary_text:
                  `Bu, "${question || "belirtilmeyen soru"}" sorusu için üretilen ` +
                  `önizleme özetidir (Tests/Presentation demo). Gerçek entegrasyonda ` +
                  `bu alan Application katmanındaki özetleme adımından gelir.`,
              },
              routing: { success: true, department: "Musteri Hizmetleri" },
              writing: { success: true, answer },
            },
          } as TData,
        });
      }, 700);
      return;
    }

    const target = resolveApiUrl(url);

    fetch(target, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(model),
    })
      .then(async (r) => {
        let data: WireResponse<TData>;
        try {
          data = (await r.json()) as WireResponse<TData>;
        } catch {
          const hint =
            r.status === 413
              ? "Dosya çok büyük (HTTP 413). Lütfen 10MB altı bir dosya deneyin."
              : `Invalid JSON from Application (HTTP ${r.status})`;
          onResult({
            Success: false,
            Message: hint,
            Code: r.status,
          });
          return;
        }

        if (!r.ok && data.Success !== true) {
          onResult({
            Success: false,
            Message: data.Message || `Application error (HTTP ${r.status})`,
            Code: data.Code ?? r.status,
            AdditionalData: data.AdditionalData,
          });
          return;
        }

        onResult(data);
      })
      .catch((err: Error) => {
        onResult({
          Success: false,
          Message: err.message || "Network error — is Application running on :8080?",
        });
      });
  },
};
