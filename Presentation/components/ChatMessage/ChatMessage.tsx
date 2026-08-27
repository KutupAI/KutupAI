import React, { useState } from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import SourceCitations from "../SourceCitations/SourceCitations";
import { SummaryIcon, SourceIcon, FileIcon } from "../icons/SectionIcons";
import { ChatMessageModel } from "../../models/ChatMessage";
import { isImageFile } from "../../utils/fileValidators";
import styles from "./ChatMessage.module.css";

interface ChatMessageProps {
  message: ChatMessageModel;
}

type DetailRow = { label: string; value: string };

const renderContent = (content: string) => {
  const blocks = content.split(/\n{2,}/);

  return blocks.map((block, blockIndex) => {
    const lines = block.split("\n").filter(Boolean);
    const isList = lines.length > 0 && lines.every((l) => /^\s*[-*]\s+/.test(l));

    if (isList) {
      return (
        <ul key={blockIndex} className={styles.list}>
          {lines.map((line, i) => (
            <li key={i}>{line.replace(/^\s*[-*]\s+/, "")}</li>
          ))}
        </ul>
      );
    }

    return (
      <p key={blockIndex} className={styles.paragraph}>
        {lines.join(" ")}
      </p>
    );
  });
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === "user";

  // Yalnızca yerel UI durumu — yeni bir istek/mesaj yok.
  const [detayOpen, setDetayOpen] = useState(false);
  const [kaynakOpen, setKaynakOpen] = useState(false);

  const state = message.pipelineState;
  const sources = message.pipelineState?.rag.rag_data?.results ?? [];
  const hasKaynak = sources.length > 0;

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
    state?.extraction.phone ? { label: "Telefon", value: state.extraction.phone } : null,
    state?.extraction.email ? { label: "E-posta", value: state.extraction.email } : null,
  ].filter((row): row is DetailRow => row !== null);
  const hasDetay = documentRows.length > 0 || extractionRows.length > 0 || hasKaynak;

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

        {/* Detay, agent'lerin çıkardığı yapılandırılmış belge bilgisini gösterir. */}
        {!isUser && hasDetay && (
          <>
            {!detayOpen && (
              <button
                type="button"
                className={styles.revealTrigger}
                onClick={() => setDetayOpen(true)}
              >
                <SummaryIcon size={14} />
                Detay
                <span className={styles.revealChevron} aria-hidden="true">▾</span>
              </button>
            )}

            {detayOpen && (
              <div className={`${styles.expandBox} ${styles.expandBoxSummary}`}>
                <div className={styles.expandBoxHeader}>
                  <SummaryIcon size={14} />
                  <span>Detay</span>
                </div>
                {documentRows.length > 0 && (
                  <section className={styles.detailSection}>
                    <h4 className={styles.detailTitle}>Belge Özeti</h4>
                    <dl className={styles.detailRows}>
                      {documentRows.map((row) => (
                        <React.Fragment key={row.label}>
                          <dt>{row.label}</dt>
                          <dd>{row.value}</dd>
                        </React.Fragment>
                      ))}
                    </dl>
                  </section>
                )}

                {extractionRows.length > 0 && (
                  <section className={styles.detailSection}>
                    <h4 className={styles.detailTitle}>Çıkarılan Bilgiler</h4>
                    <dl className={styles.detailRows}>
                      {extractionRows.map((row) => (
                        <React.Fragment key={row.label}>
                          <dt>{row.label}</dt>
                          <dd>{row.value}</dd>
                        </React.Fragment>
                      ))}
                    </dl>
                  </section>
                )}

                {/* --- Kaynak: sadece Detay açıldıktan SONRA, ve sadece
                    istenirse (varsayılan gizli) görünür. --- */}
                {hasKaynak && !kaynakOpen && (
                  <button
                    type="button"
                    className={styles.revealTriggerInline}
                    onClick={() => setKaynakOpen(true)}
                  >
                    <SourceIcon size={14} />
                    Kaynak
                    <span className={styles.revealChevron} aria-hidden="true">▾</span>
                  </button>
                )}

                {hasKaynak && kaynakOpen && (
                  <div className={`${styles.expandBox} ${styles.expandBoxSources}`}>
                    <div className={styles.expandBoxHeader}>
                      <SourceIcon size={14} />
                      <span>Kaynak</span>
                    </div>
                    <SourceCitations results={sources} bare />
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
