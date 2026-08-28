import React, { useState } from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import SourceCitations from "../SourceCitations/SourceCitations";
import { SummaryIcon, SourceIcon, FileIcon } from "../icons/SectionIcons";
import { ChatMessageModel } from "../../models/ChatMessage";
import { isImageFile } from "../../utils/fileValidators";
import styles from "./ChatMessage.module.css";

interface ChatMessageProps {
  message: ChatMessageModel;
  isAdmin: boolean;
}

type DetailRow = { label: string; value: string };

// Writer'ın kullandığı basit Markdown kalın vurgusunu arayüzde okunabilir
// gösterir. Tam bir Markdown motoruna gerek duymadan `**metin**` kalıbını
// güvenli React düğümlerine dönüştürür.
const renderInline = (text: string) =>
  text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={index}>{part.slice(2, -2)}</strong>
    ) : (
      <React.Fragment key={index}>{part}</React.Fragment>
    )
  );

const renderContent = (content: string) => {
  const blocks = content.split(/\n{2,}/);

  return blocks.map((block, blockIndex) => {
    const lines = block.split("\n").filter(Boolean);
    const isList = lines.length > 0 && lines.every((l) => /^\s*[-*]\s+/.test(l));

    if (isList) {
      return (
        <ul key={blockIndex} className={styles.list}>
          {lines.map((line, i) => (
            <li key={i}>{renderInline(line.replace(/^\s*[-*]\s+/, ""))}</li>
          ))}
        </ul>
      );
    }

    return (
      <p key={blockIndex} className={styles.paragraph}>
        {renderInline(lines.join(" "))}
      </p>
    );
  });
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isAdmin }) => {
  const isUser = message.role === "user";

  // Yalnızca yerel UI durumu — yeni bir istek/mesaj yok.
  const [ozetOpen, setOzetOpen] = useState(false);
  const [kaynakOpen, setKaynakOpen] = useState(false);

  const state = message.pipelineState;
  const sources = message.pipelineState?.rag.rag_data?.results ?? [];
  const hasKaynak = sources.length > 0;
  // T.C. Kimlik No yalnızca kendi alan etiketiyle birlikte ve 11 hane ise
  // gösterilir. Böylece 11 haneli bir telefon numarası yanlışlıkla T.C.
  // olarak yorumlanmaz; veri mevcut OCR metninden okunur.
  const tcKimlikNoFromOcr = state?.ocr.ocr_data?.full_text.match(
    /T\.?\s*C\.?\s*(?:Kimlik\s*)?(?:No|Numarası|Numarasi)?\s*[:\-]?\s*(\d{11})(?!\d)/iu
  )?.[1] ?? null;
  // Bazı çıkarım sonuçlarında 11 haneli T.C. numarası yanlışlıkla telefon
  // alanına gelebilir. OCR'da T.C. etiketi varsa, arayüz bunu doğru alanda
  // gösterir ve aynı değeri telefon olarak tekrar etmez.
  const phoneValue = state?.extraction.phone ?? null;
  const looksLikeTc = /^\d{11}$/.test(phoneValue?.replace(/\D/g, "") ?? "");
  const hasTcLabel = /T\.?\s*C\.?\s*(?:Kimlik\s*)?(?:No|Numarası|Numarasi)?/iu.test(
    state?.ocr.ocr_data?.full_text ?? ""
  );
  const tcKimlikNo = tcKimlikNoFromOcr ?? (hasTcLabel && looksLikeTc ? phoneValue?.replace(/\D/g, "") ?? null : null);
  const phoneIsTc = Boolean(tcKimlikNo && phoneValue?.replace(/\D/g, "") === tcKimlikNo);

  const documentRows: DetailRow[] = [
    state?.classification.document_type
      ? { label: "Belge türü", value: state.classification.document_type }
      : null,
    state?.ocr.ocr_data
      ? {
          label: "İmza",
          value: state.ocr.ocr_data.vision.signature.detected ? "Tespit edildi" : "Tespit edilmedi",
        }
      : null,
    state?.routing.department ? { label: "Yönlendirme", value: state.routing.department } : null,
  ].filter((row): row is DetailRow => row !== null);

  const extractionRows: DetailRow[] = [
    state?.extraction.sender ? { label: "Gönderen", value: state.extraction.sender } : null,
    state?.extraction.date ? { label: "Tarih", value: state.extraction.date } : null,
    state?.extraction.address ? { label: "Adres", value: state.extraction.address } : null,
    state?.extraction.phone && !phoneIsTc ? { label: "Telefon", value: state.extraction.phone } : null,
    state?.extraction.email ? { label: "E-posta", value: state.extraction.email } : null,
    tcKimlikNo ? { label: "T.C. Kimlik No", value: tcKimlikNo } : null,
  ].filter((row): row is DetailRow => row !== null);

  const processSteps: DetailRow[] = [
    { label: "Belge okundu", value: state?.ocr.success ? "Tamamlandı" : "Tamamlanamadı" },
    { label: "Bilgiler çıkarıldı", value: state?.extraction.success ? "Tamamlandı" : "Tamamlanamadı" },
    { label: "Mevzuat araştırıldı", value: state?.rag.success ? "Tamamlandı" : "Tamamlanamadı" },
    { label: "Yanıt hazırlandı", value: state?.writing.success ? "Tamamlandı" : "Tamamlanamadı" },
  ];
  const hasOzet = documentRows.length > 0 || extractionRows.length > 0 || Boolean(state?.summary.rag_summary_text.trim());

  return (
    <div
      className={`${styles.row} ${isUser ? styles.rowUser : styles.rowAssistant}`}
    >
      {!isUser && (
        <div className={styles.avatar} aria-hidden="true">
          <PoleStarIcon size={18} />
        </div>
      )}

      <div className={styles.bubbleCol}>
        <div
          className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleAssistant}`}
        >
          {!isUser && <div className={styles.senderLabel}>KutupAI</div>}

          {message.file && (
            <div className={styles.attachedFile}>
              {isImageFile(message.file.name) && message.file.previewUrl ? (
                <img
                  src={message.file.previewUrl}
                  alt={message.file.name}
                  className={styles.attachedThumb}
                />
              ) : (
                <span className={styles.attachedIcon} aria-hidden="true">
                  <FileIcon size={16} />
                </span>
              )}
              <span className={styles.attachedName}>{message.file.name}</span>
            </div>
          )}

          <div className={styles.content}>{renderContent(message.content)}</div>

          {message.status === "error" && (
            <div className={styles.errorTag} role="alert">
              Gönderilemedi
            </div>
          )}
        </div>

        {/* Kaynak herkes için görünür; alıntı metni yalnızca istenince açılır. */}
        {!isUser && hasKaynak && (
          <>
            {!kaynakOpen && (
              <button
                type="button"
                className={styles.revealTrigger}
                onClick={() => setKaynakOpen(true)}
              >
                <SourceIcon size={14} />
                Kaynak
                <span className={styles.revealChevron} aria-hidden="true">▾</span>
              </button>
            )}

            {kaynakOpen && (
              <div className={`${styles.expandBox} ${styles.expandBoxSources}`}>
                <div className={styles.expandBoxHeader}>
                  <SourceIcon size={14} />
                  <span>Kaynak</span>
                </div>
                <SourceCitations results={sources} bare />
              </div>
            )}
          </>
        )}

        {/* Özet yalnızca yerel yönetici oturumunda açılabilir. */}
        {!isUser && isAdmin && hasOzet && (
          <>
            {!ozetOpen && (
              <button type="button" className={`${styles.revealTrigger} ${styles.adminTrigger}`} onClick={() => setOzetOpen(true)}>
                <SummaryIcon size={14} />
                Özet
                <span className={styles.revealChevron} aria-hidden="true">▾</span>
              </button>
            )}

            {ozetOpen && (
              <div className={`${styles.expandBox} ${styles.expandBoxSummary}`}>
                <div className={styles.expandBoxHeader}>
                  <SummaryIcon size={14} />
                  <span>Özet</span>
                </div>

                {state?.summary.rag_summary_text.trim() && (
                  <p className={styles.summaryText}>{state.summary.rag_summary_text}</p>
                )}

                {documentRows.length > 0 && (
                  <section className={styles.detailSection}>
                    <h4 className={styles.detailTitle}>Belge Bilgileri</h4>
                    <dl className={styles.detailRows}>
                      {documentRows.map((row) => (
                        <React.Fragment key={row.label}><dt>{row.label}</dt><dd>{row.value}</dd></React.Fragment>
                      ))}
                    </dl>
                  </section>
                )}

                {extractionRows.length > 0 && (
                  <section className={styles.detailSection}>
                    <h4 className={styles.detailTitle}>Çıkarılan Bilgiler</h4>
                    <dl className={styles.detailRows}>
                      {extractionRows.map((row) => (
                        <React.Fragment key={row.label}><dt>{row.label}</dt><dd>{row.value}</dd></React.Fragment>
                      ))}
                    </dl>
                  </section>
                )}

                <section className={styles.processSection}>
                  <h4 className={styles.detailTitle}>İşlem Adımları</h4>
                  <ol className={styles.processList}>
                    {processSteps.map((step) => (
                      <li key={step.label} className={step.value === "Tamamlandı" ? styles.processDone : styles.processFailed}>
                        <span className={styles.processMark} aria-hidden="true">{step.value === "Tamamlandı" ? "✓" : "!"}</span>
                        <span>{step.label}</span><small>{step.value}</small>
                      </li>
                    ))}
                  </ol>
                </section>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
