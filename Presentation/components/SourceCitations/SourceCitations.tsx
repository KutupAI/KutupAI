import React, { useState } from "react";
import { RagResultItem } from "../../models/AnalysisResult";
import { formatArticleLabel, formatLawTitle, formatPageRange } from "../../utils/formatters";
import styles from "./SourceCitations.module.css";

interface SourceCitationsProps {
  results: RagResultItem[];
  /** true iken kendi <section>/başlık zarfını çizmez -- sadece liste;
   *  AnalysisResultCard artık "Kaynak" başlığını kendi sectionCard
   *  kutusunda (ikon + sayaç ile) çiziyor, burada tekrar etmesin diye. */
  bare?: boolean;
}

/**
 * "Kaynak" (Source) — yanıtın hangi kanun/madde'ye dayandığını gösteren
 * profesyonel alıntı listesi. Amaç: cevaba güveni artırmak, kullanıcının
 * "bu bilgi nereden geldi?" sorusuna doğrudan cevap vermek.
 *
 * Her kaynak numaralandırılmış bir kart olarak gösterilir (kanun başlığı +
 * kanun no + madde no + sayfa), tıklanınca kısa alıntı metnini açar/kapar
 * (snippet varsa) -- ham JSON veya chunk_id gibi teknik detaylar asla
 * kullanıcıya gösterilmez, sadece okunabilir hukuki referans.
 */
const SourceCitations: React.FC<SourceCitationsProps> = ({ results, bare = false }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  if (results.length === 0) return null;

  const list = (
    <ol className={styles.list}>
        {results.map((r, i) => {
          const title = formatLawTitle(r.source_file);
          const article = formatArticleLabel(r.article_no);
          const pages = formatPageRange(r.page_start, r.page_end);
          const key = r.chunk_id ?? `${title}-${i}`;
          const isExpanded = expandedId === key;
          const canExpand = Boolean(r.snippet);

          return (
            <li key={key} className={styles.item}>
              <button
                type="button"
                className={styles.card}
                onClick={() => canExpand && setExpandedId(isExpanded ? null : key)}
                aria-expanded={canExpand ? isExpanded : undefined}
                disabled={!canExpand}
              >
                <span className={styles.badgeNum} aria-hidden="true">
                  {i + 1}
                </span>

                <span className={styles.cardBody}>
                  <span className={styles.cardTitle}>{title}</span>
                  <span className={styles.cardMeta}>
                    {r.law_number && <span>Kanun No {r.law_number}</span>}
                    {article && <span>{article}</span>}
                    {pages && <span>{pages}</span>}
                  </span>
                </span>

                {canExpand && (
                  <span className={styles.chevron} aria-hidden="true">
                    {isExpanded ? "−" : "+"}
                  </span>
                )}
              </button>

              {isExpanded && r.snippet && (
                <blockquote className={styles.snippet}>{r.snippet}</blockquote>
              )}
            </li>
          );
        })}
    </ol>
  );

  if (bare) return list;

  return (
    <section className={styles.root} aria-label="Kaynaklar">
      <h3 className={styles.heading}>
        Kaynak
        <span className={styles.count}>{results.length}</span>
      </h3>
      {list}
    </section>
  );
};

export default SourceCitations;