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

  // Yalnızca yerel UI durumu — yeni bir istek/mesaj YOK. Aynı pipelineState
  // içindeki veri (özet metni + kaynaklar) burada, mesajın altında açılır
  // bir dikdörtgen olarak gösterilir.
  const [detayOpen, setDetayOpen] = useState(false);
  const [kaynakOpen, setKaynakOpen] = useState(false);

  const summaryText = message.pipelineState?.summary.rag_summary_text?.trim() ?? "";
  const sources = message.pipelineState?.rag.rag_data?.results ?? [];
  const hasDetay = summaryText.length > 0;
  const hasKaynak = sources.length > 0;

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

        {/* --- Detay: Yanıt'ın hemen altında bir düğme; tıklanınca YERİNDE
            açılan bir dikdörtgen (yeni mesaj/istek yok). --- */}
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
                <div className={styles.expandBoxBody}>{renderContent(summaryText)}</div>

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