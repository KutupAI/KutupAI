import React, { useState } from "react";
import {
  PipelineState,
  toStructuredDisplay,
} from "../../models/AnalysisResult";
import StatusBadge from "../StatusBadge/StatusBadge";
import SourceCitations from "../SourceCitations/SourceCitations";
import { AnswerIcon, SummaryIcon, SourceIcon, DocumentIcon } from "../icons/SectionIcons";
import styles from "./AnalysisResultCard.module.css";

interface AnalysisResultCardProps {
  state: PipelineState;
}

const EMPTY = "—";

type ExpandablePanel = "detailed" | null;


const CardHeader: React.FC<{ icon: React.ReactNode; title: string; badge?: string | number }> = ({
  icon,
  title,
  badge,
}) => (
  <div className={styles.cardHeader}>
    <span className={styles.cardIcon} aria-hidden="true">
      {icon}
    </span>
    <h3 className={styles.cardTitle}>{title}</h3>
    {badge !== undefined && <span className={styles.cardBadge}>{badge}</span>}
  </div>
);

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

const renderParagraphs = (text: string, className: string) =>
  text.split(/\n{2,}/).map((paragraph, i) => (
    <p key={i} className={className}>
      {paragraph.split("\n").map((line, j, arr) => (
        <React.Fragment key={j}>
          {line}
          {j < arr.length - 1 && <br />}
        </React.Fragment>
      ))}
    </p>
  ));

/**
 * Structured Explanatory Style — v2 (ayrı kutular).
 *
 * Yerleşim: Yanıt / Özet / Kaynak artık HER ZAMAN görünür, birbirinden
 * tamamen ayrı, kendi kenarlıklı kutusu (sectionCard) ve kendi renk
 * vurgusu olan üç ayrı dikdörtgen olarak art arda dizilir -- eskisi gibi
 * "Özet" bir akordeon düğmesinin arkasında saklanmıyor, bu yüzden hiçbir
 * zaman "çalışmıyor" gibi görünmüyor (içerik boşsa bile kutusu görünür,
 * boş-durum mesajı gösterir). Belge Bilgileri / Çıkarılan Bilgiler de
 * kendi kutularında. Sadece teknik "Detaylı Bilgiler" panosu hâlâ
 * açılır-kapanır (gürültüyü azaltmak için).
 */
const AnalysisResultCard: React.FC<AnalysisResultCardProps> = ({ state }) => {
  const [expanded, setExpanded] = useState<ExpandablePanel>(null);

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

  const hasSummary = state.summary.rag_summary_text.trim().length > 0;
  const confidencePct =
    state.classification.classification_confidence !== null
      ? `%${Math.round(state.classification.classification_confidence * 100)}`
      : null;

  const toggle = (panel: ExpandablePanel) => {
    setExpanded((current) => (current === panel ? null : panel));
  };

  return (
    <article className={styles.root} aria-label="Yapılandırılmış yanıt">
      {/* --- 1) Yanıt: ana cevap, her zaman kendi kutusunda --- */}
      {answer?.trim() && (
        <section className={`${styles.sectionCard} ${styles.cardAnswer}`}>
          <CardHeader icon={<AnswerIcon />} title="Yanıt" />
          <div className={styles.answerBody}>
            {renderParagraphs(answer, styles.answerParagraph)}
          </div>
        </section>
      )}

      {/* --- 2) Özet: kendi ayrı kutusu, HER ZAMAN görünür (artık bir
          düğmenin arkasında saklanmıyor). İçerik boşsa nazik bir boş-durum
          mesajı gösterir; buton hiçbir zaman devre dışı bırakılmaz. --- */}
      <section className={`${styles.sectionCard} ${styles.cardSummary}`}>
        <CardHeader icon={<SummaryIcon />} title="Özet" />
        {hasSummary ? (
          <div className={styles.answerBody}>
            {renderParagraphs(state.summary.rag_summary_text, styles.answerParagraph)}
          </div>
        ) : (
          <p className={styles.emptyState}>Bu yanıt için ayrı bir özet üretilmedi.</p>
        )}
      </section>

      {/* --- 3) Kaynak: kendi ayrı kutusu --- */}
      {state.rag.rag_data && state.rag.rag_data.results.length > 0 && (
        <section className={`${styles.sectionCard} ${styles.cardSources}`}>
          <CardHeader icon={<SourceIcon />} title="Kaynak" badge={state.rag.rag_data.results.length} />
          <SourceCitations results={state.rag.rag_data.results} bare />
        </section>
      )}

      {/* --- Belge Bilgileri (eskiden "Belge Özeti" — "Özet" ile karışmaması
          için yeniden adlandırıldı) ve Çıkarılan Bilgiler: yardımcı, ayrı
          kutular. --- */}
      <section className={`${styles.sectionCard} ${styles.cardNeutral}`}>
        <CardHeader icon={<DocumentIcon />} title="Belge Bilgileri" />
        <dl className={styles.factList}>
          <FactRow label="Belge türü" value={document_type} />
          <FactRow label="İmza" value={signatureText} />
          <FactRow label="Yönlendirme" value={department} />
        </dl>
      </section>

      <section className={`${styles.sectionCard} ${styles.cardNeutral}`}>
        <CardHeader icon={<DocumentIcon />} title="Çıkarılan Bilgiler" />
        <dl className={styles.factList}>
          <FactRow label="Gönderen" value={extraction.sender} />
          <FactRow label="Tarih" value={extraction.date} />
          <FactRow label="Adres" value={extraction.address} />
          <FactRow label="Telefon" value={extraction.phone} />
          <FactRow label="E-posta" value={extraction.email} />
        </dl>
      </section>

      {/* --- Detaylı: teknik pipeline detayları, tek açılır-kapanır düğme --- */}
      <div className={styles.expandBar}>
        <button
          type="button"
          className={`${styles.expandBtn} ${expanded === "detailed" ? styles.expandBtnActive : ""}`}
          onClick={() => toggle("detailed")}
          aria-expanded={expanded === "detailed"}
        >
          Detaylı Bilgiler
          <span className={styles.chev} aria-hidden="true">{expanded === "detailed" ? "▲" : "▼"}</span>
        </button>
      </div>

      {expanded === "detailed" && (
        <div className={styles.expandPanel}>
          <section className={styles.detailBlock}>
            <h3 className={styles.sectionHeading}>Talep</h3>
            <dl className={styles.factList}>
              <FactRow label="Soru" value={state.request.question} />
              <FactRow label="Belge adı" value={state.request.document?.file_name} />
              <FactRow label="Dosya türü" value={state.request.document?.file_type} />
            </dl>
          </section>

          <section className={styles.detailBlock}>
            <h3 className={styles.sectionHeading}>OCR</h3>
            <div className={styles.badgeRow}>
              <StatusBadge
                label={state.ocr.success ? "Başarılı" : "Başarısız"}
                tone={state.ocr.success ? "positive" : "neutral"}
              />
              {state.ocr.ocr_data?.vision.stamp.detected && (
                <StatusBadge label="Kaşe tespit edildi" tone="positive" />
              )}
            </div>
            <dl className={styles.factList}>
              <FactRow label="Sayfa sayısı" value={state.ocr.ocr_data?.page_count?.toString()} />
              <FactRow label="Dil" value={state.ocr.ocr_data?.language} />
              <FactRow label="İmza" value={signatureText} />
            </dl>
          </section>

          <section className={styles.detailBlock}>
            <h3 className={styles.sectionHeading}>Sınıflandırma</h3>
            <dl className={styles.factList}>
              <FactRow label="Belge türü" value={state.classification.document_type} />
              <FactRow label="Güven skoru" value={confidencePct} />
            </dl>
          </section>

          <section className={styles.detailBlock}>
            <h3 className={styles.sectionHeading}>Doğrulama</h3>
            <div className={styles.badgeRow}>
              <StatusBadge
                label={state.validation.is_complete ? "Eksiksiz" : "Eksik bilgi var"}
                tone={state.validation.is_complete ? "positive" : "neutral"}
              />
            </div>
            {state.validation.errors.length > 0 && (
              <ul className={styles.issueList}>
                {state.validation.errors.map((e, i) => (
                  <li key={i} className={styles.issueError}>{e}</li>
                ))}
              </ul>
            )}
            {state.validation.warnings.length > 0 && (
              <ul className={styles.issueList}>
                {state.validation.warnings.map((w, i) => (
                  <li key={i} className={styles.issueWarning}>{w}</li>
                ))}
              </ul>
            )}
          </section>

          {state.rag.rag_data && (
            <section className={styles.detailBlock}>
              <h3 className={styles.sectionHeading}>RAG (Mevzuat Araması)</h3>
              <dl className={styles.factList}>
                <FactRow label="İşlem" value={state.rag.rag_data.operation} />
                <FactRow label="Sorgu" value={state.rag.rag_data.query} />
              </dl>
              {state.rag.rag_data.results.length > 0 && (
                <SourceCitations results={state.rag.rag_data.results} />
              )}
            </section>
          )}

          <section className={styles.detailBlock}>
            <h3 className={styles.sectionHeading}>Yönlendirme</h3>
            <dl className={styles.factList}>
              <FactRow label="Birim" value={state.routing.department} />
            </dl>
          </section>
        </div>
      )}
    </article>
  );
};

export default AnalysisResultCard;