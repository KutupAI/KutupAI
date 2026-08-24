import React from "react";
import {
  PipelineState,
  toStructuredDisplay,
} from "../../models/AnalysisResult";
import styles from "./AnalysisResultCard.module.css";

interface AnalysisResultCardProps {
  state: PipelineState;
}

const EMPTY = "—";

const FactRow: React.FC<{ label: string; value: string | null | undefined }> = ({
  label,
  value,
}) => (
  <div className={styles.factRow}>
    <dt className={styles.factLabel}>{label}</dt>
    <dd className={value ? styles.factValue : styles.factValueMuted}>
      {value?.trim() ? value : EMPTY}
    </dd>
  </div>
);

/**
 * Structured Explanatory Style — surfaces ONLY the presentation subset:
 * signature, document_type, extraction fields, department, answer.
 * Full pipeline JSON is never rendered.
 */
const AnalysisResultCard: React.FC<AnalysisResultCardProps> = ({ state }) => {
  const {
    signature,
    document_type,
    extraction,
    department,
    answer,
  } = toStructuredDisplay(state);

  const signatureText = signature
    ? signature.detected
      ? signature.handwritten
        ? "Tespit edildi (el yazısı)"
        : "Tespit edildi"
      : "Tespit edilmedi"
    : null;

  return (
    <article className={styles.root} aria-label="Yapılandırılmış yanıt">
      {answer?.trim() && (
        <section className={styles.answerSection}>
          <h3 className={styles.sectionHeading}>Yanıt</h3>
          <div className={styles.answerBody}>
            {answer.split(/\n{2,}/).map((paragraph, i) => (
              <p key={i} className={styles.answerParagraph}>
                {paragraph.split("\n").map((line, j, arr) => (
                  <React.Fragment key={j}>
                    {line}
                    {j < arr.length - 1 && <br />}
                  </React.Fragment>
                ))}
              </p>
            ))}
          </div>
        </section>
      )}

      <section className={styles.factsSection}>
        <h3 className={styles.sectionHeading}>Belge Özeti</h3>
        <dl className={styles.factList}>
          <FactRow label="Belge türü" value={document_type} />
          <FactRow label="İmza" value={signatureText} />
          <FactRow label="Yönlendirme" value={department} />
        </dl>
      </section>

      <section className={styles.extractionSection}>
        <h3 className={styles.sectionHeading}>Çıkarılan Bilgiler</h3>
        <dl className={styles.factList}>
          <FactRow label="Gönderen" value={extraction.sender} />
          <FactRow label="Tarih" value={extraction.date} />
          <FactRow label="Adres" value={extraction.address} />
          <FactRow label="Telefon" value={extraction.phone} />
          <FactRow label="E-posta" value={extraction.email} />
        </dl>
      </section>
    </article>
  );
};

export default AnalysisResultCard;
