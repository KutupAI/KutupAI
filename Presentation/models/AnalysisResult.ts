// AnalysisResult
// Unified pipeline state returned by the Application layer for a document
// analysis / chat turn. Each agent/layer writes ONLY to its own top-level
// section. This exact 9-key shape must be preserved end-to-end:
//
//   {
//     "request": {},
//     "ocr": {},
//     "classification": {},
//     "extraction": {},
//     "validation": {},
//     "rag": {},
//     "summary": {},
//     "routing": {},
//     "writing": {}
//   }
//
// Presentation displays ONLY a structured subset of the filled contract:
// signature, document_type, extraction fields, department, answer.

export interface PipelineState {
  request: RequestSection;
  ocr: OcrSection;
  classification: ClassificationSection;
  extraction: ExtractionSection;
  validation: ValidationSection;
  rag: RagSection;
  summary: SummarySection;
  routing: RoutingSection;
  writing: WritingSection;
}

export interface RequestSection {
  success: boolean;
  question: string;
  document: {
    document_id: string;
    file_name: string;
    file_type: string;
  } | null;
}

export interface OcrSignature {
  detected: boolean;
  handwritten: boolean;
}

export interface OcrStamp {
  detected: boolean;
}

export interface OcrSection {
  success: boolean;
  ocr_data: {
    page_count: number;
    language: string;
    pages: unknown[];
    full_text: string;
    vision: {
      signature: OcrSignature;
      stamp: OcrStamp;
    };
  } | null;
}

export interface ClassificationSection {
  success: boolean;
  document_type: string | null;
  classification_confidence: number | null;
}

export interface ExtractionSection {
  success: boolean;
  sender: string | null;
  date: string | null;
  address: string | null;
  phone: string | null;
  email: string | null;
}

export interface ValidationSection {
  success: boolean;
  is_complete: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Bir hukuki kaynak alıntısı (RAG retriever'ın bulduğu kanun maddesi).
 * Şu alanlar gerçek RAG veri setinden doğrulandı (indexed_chunks.json /
 * facts_registry.json / amendment_ledger.json şeması): law_number,
 * article_no, source_file, page_start/page_end. `snippet` ve `score`
 * opsiyonel -- retriever her zaman dönmeyebilir, arayüz ikisi de
 * olmadan da düzgün çalışır.
 */
export interface RagResultItem {
  chunk_id: string | null;
  law_number: string | null;
  article_no: string | null;
  /** Ham dosya adı, örn. "1076_Yedek Subaylar Kanunu.pdf" -- görüntüleme
   *  için utils/formatters.ts::formatLawTitle ile temizlenir. */
  source_file: string | null;
  page_start: number | null;
  page_end: number | null;
  snippet: string | null;
  score: number | null;
}

export interface RagSection {
  success: boolean;
  rag_data: {
    operation: string;
    query: string;
    results: RagResultItem[];
  } | null;
}

export interface SummarySection {
  success: boolean;
  rag_summary_text: string;
}

export interface RoutingSection {
  success: boolean;
  department: string | null;
}

export interface WritingSection {
  success: boolean;
  answer: string;
}

/** Empty envelope — same shape Application/Orchestration start with. */
export const createEmptyPipelineState = (): PipelineState => ({
  request: { success: false, question: "", document: null },
  ocr: { success: false, ocr_data: null },
  classification: { success: false, document_type: null, classification_confidence: null },
  extraction: { success: false, sender: null, date: null, address: null, phone: null, email: null },
  validation: { success: false, is_complete: false, errors: [], warnings: [] },
  rag: { success: false, rag_data: null },
  summary: { success: false, rag_summary_text: "" },
  routing: { success: false, department: null },
  writing: { success: false, answer: "" },
});

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const asBool = (v: unknown, fallback = false): boolean =>
  typeof v === "boolean" ? v : fallback;

const asStr = (v: unknown, fallback = ""): string =>
  typeof v === "string" ? v : fallback;

const asStrOrNull = (v: unknown): string | null =>
  typeof v === "string" ? v : v === null || v === undefined ? null : String(v);

const asNumOrNull = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Bir tek RAG sonucunu (kanun maddesi kanıtı) esnek şekilde parse eder --
 *  retriever farklı sürümlerde farklı alan adları kullanmış olabilir
 *  (örn. "text" / "snippet" / "full_text"), bu yüzden birkaç olası ad
 *  denenir; hiçbiri yoksa alan sessizce null kalır (arayüz çökmez). */
const asRagResultItem = (v: unknown): RagResultItem | null => {
  if (!isRecord(v)) return null;
  const pick = (...keys: string[]): unknown => {
    for (const k of keys) if (v[k] !== undefined) return v[k];
    return undefined;
  };
  return {
    chunk_id: asStrOrNull(pick("chunk_id")),
    law_number: asStrOrNull(pick("law_number", "kanun_no")),
    article_no: asStrOrNull(pick("article_no", "madde_no")),
    source_file: asStrOrNull(pick("source_file", "kaynak_dosya")),
    page_start: asNumOrNull(pick("page_start", "kaynak_sayfa")),
    page_end: asNumOrNull(pick("page_end")),
    snippet: asStrOrNull(pick("snippet", "text", "full_text", "evidence_text")),
    score: asNumOrNull(pick("score", "confidence", "relevance")),
  };
};

/**
 * Normalize a raw API / demo payload into the canonical 9-key PipelineState.
 * Accepts:
 *  - Presentation State envelope { request, ocr, … writing }
 *  - Orchestration document (Data[0]) with stage keys nested on the doc
 * Empty `{}` stage sections are treated as empty defaults.
 */
export const normalizePipelineState = (raw: unknown): PipelineState => {
  const empty = createEmptyPipelineState();
  if (!isRecord(raw)) return empty;

  // Orchestration Data[0] shape → lift into envelope first.
  const envelope: Record<string, unknown> =
    isRecord(raw.request) || ("ocr" in raw && !("document_id" in raw))
      ? raw
      : {
          request: {
            success: true,
            question: asStr(raw.question),
            document: {
              document_id: asStr(raw.document_id),
              file_name: asStr(raw.file_name),
              file_type: asStr(raw.file_type),
            },
          },
          ocr: isRecord(raw.ocr) ? raw.ocr : {},
          classification: isRecord(raw.classification) ? raw.classification : {},
          extraction: isRecord(raw.extraction) ? raw.extraction : {},
          validation: isRecord(raw.validation) ? raw.validation : {},
          rag: isRecord(raw.rag) ? raw.rag : {},
          summary:
            isRecord(raw.summary)
              ? raw.summary
              : typeof raw.summary === "string" && raw.summary
                ? { success: true, rag_summary_text: raw.summary }
                : {},
          routing: isRecord(raw.routing) ? raw.routing : {},
          writing:
            isRecord(raw.writing)
              ? raw.writing
              : typeof raw.answer === "string" && raw.answer
                ? { success: true, answer: raw.answer }
                : {},
        };

  const requestRaw = isRecord(envelope.request) ? envelope.request : {};
  const docRaw = isRecord(requestRaw.document) ? requestRaw.document : null;

  const ocrRaw = isRecord(envelope.ocr) ? envelope.ocr : {};
  const ocrDataRaw = isRecord(ocrRaw.ocr_data) ? ocrRaw.ocr_data : null;
  const visionRaw =
    ocrDataRaw && isRecord(ocrDataRaw.vision) ? ocrDataRaw.vision : null;
  const signatureRaw =
    visionRaw && isRecord(visionRaw.signature) ? visionRaw.signature : null;
  const stampRaw =
    visionRaw && isRecord(visionRaw.stamp) ? visionRaw.stamp : null;

  const classificationRaw = isRecord(envelope.classification)
    ? envelope.classification
    : {};
  const extractionRaw = isRecord(envelope.extraction) ? envelope.extraction : {};
  const validationRaw = isRecord(envelope.validation) ? envelope.validation : {};
  const ragRaw = isRecord(envelope.rag) ? envelope.rag : {};
  const ragDataRaw = isRecord(ragRaw.rag_data) ? ragRaw.rag_data : null;
  const summaryRaw = isRecord(envelope.summary) ? envelope.summary : {};
  const routingRaw = isRecord(envelope.routing) ? envelope.routing : {};
  const writingRaw = isRecord(envelope.writing) ? envelope.writing : {};

  return {
    request: {
      success: asBool(requestRaw.success, empty.request.success),
      question: asStr(requestRaw.question, empty.request.question),
      document: docRaw
        ? {
            document_id: asStr(docRaw.document_id),
            file_name: asStr(docRaw.file_name),
            file_type: asStr(docRaw.file_type),
          }
        : null,
    },
    ocr: {
      success: asBool(ocrRaw.success, empty.ocr.success),
      ocr_data: ocrDataRaw
        ? {
            page_count: typeof ocrDataRaw.page_count === "number" ? ocrDataRaw.page_count : 0,
            language: asStr(ocrDataRaw.language, "tr"),
            pages: Array.isArray(ocrDataRaw.pages) ? ocrDataRaw.pages : [],
            full_text: asStr(ocrDataRaw.full_text),
            vision: {
              signature: {
                detected: asBool(signatureRaw?.detected),
                handwritten: asBool(signatureRaw?.handwritten),
              },
              stamp: {
                detected: asBool(stampRaw?.detected),
              },
            },
          }
        : null,
    },
    classification: {
      success: asBool(classificationRaw.success, empty.classification.success),
      document_type: asStrOrNull(classificationRaw.document_type),
      classification_confidence: asNumOrNull(classificationRaw.classification_confidence),
    },
    extraction: {
      success: asBool(extractionRaw.success, empty.extraction.success),
      sender: asStrOrNull(extractionRaw.sender),
      date: asStrOrNull(extractionRaw.date),
      address: asStrOrNull(extractionRaw.address),
      phone: asStrOrNull(extractionRaw.phone),
      email: asStrOrNull(extractionRaw.email),
    },
    validation: {
      success: asBool(validationRaw.success, empty.validation.success),
      is_complete: asBool(validationRaw.is_complete),
      errors: Array.isArray(validationRaw.errors)
        ? validationRaw.errors.map(String)
        : [],
      warnings: Array.isArray(validationRaw.warnings)
        ? validationRaw.warnings.map(String)
        : [],
    },
    rag: {
      success: asBool(ragRaw.success, empty.rag.success),
      rag_data: ragDataRaw
        ? {
            operation: asStr(ragDataRaw.operation, "retrieve"),
            query: asStr(ragDataRaw.query),
            results: Array.isArray(ragDataRaw.results)
              ? ragDataRaw.results.map(asRagResultItem).filter((r): r is RagResultItem => r !== null)
              : [],
          }
        : null,
    },
    summary: {
      success: asBool(summaryRaw.success, empty.summary.success),
      rag_summary_text: asStr(summaryRaw.rag_summary_text),
    },
    routing: {
      success: asBool(routingRaw.success, empty.routing.success),
      department: asStrOrNull(routingRaw.department),
    },
    writing: {
      success: asBool(writingRaw.success, empty.writing.success),
      answer: asStr(writingRaw.answer),
    },
  };
};

/** Fields the Structured Response UI is allowed to surface. */
export interface StructuredDisplayModel {
  signature: OcrSignature | null;
  document_type: string | null;
  extraction: Pick<ExtractionSection, "sender" | "date" | "address" | "phone" | "email">;
  department: string | null;
  answer: string;
}

export const toStructuredDisplay = (state: PipelineState): StructuredDisplayModel => ({
  signature: state.ocr.ocr_data?.vision?.signature ?? null,
  document_type: state.classification.document_type,
  extraction: {
    sender: state.extraction.sender,
    date: state.extraction.date,
    address: state.extraction.address,
    phone: state.extraction.phone,
    email: state.extraction.email,
  },
  department: state.routing.department,
  // Prefer writing.answer; fall back to summary so a skipped writer still shows text.
  answer: state.writing.answer?.trim()
    || state.summary.rag_summary_text?.trim()
    || "",
});
