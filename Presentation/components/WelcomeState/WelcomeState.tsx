import React from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import { AnswerIcon, SourceIcon, CompassIcon } from "../icons/SectionIcons";
import { CHAT_CONFIG } from "../../config/chatConfig";
import styles from "./WelcomeState.module.css";

/**
 * Her öneri kendi ikonuyla eşleştirilir (AnswerIcon yerine burada
 * "özetleme" niyetini taşıdığı için SummaryIcon değil, doğrudan konuya
 * uygun bir simge seçildi: özet → Answer/liste hissi, mevzuat → kitap,
 * yönlendirme → pusula/"Kutup Yıldızı" teması).
 */
const SUGGESTIONS: { text: string; icon: React.ReactNode }[] = [
  { text: "Bu belgeyi özetler misin?", icon: <AnswerIcon size={18} /> },
  { text: "Bu başvuru için hangi mevzuat geçerli?", icon: <SourceIcon size={18} /> },
  { text: "Bu evrağı ilgili birime nasıl yönlendiririm?", icon: <CompassIcon size={18} /> },
];

interface WelcomeStateProps {
  onSuggestionClick: (text: string) => void;
}

const WelcomeState: React.FC<WelcomeStateProps> = ({ onSuggestionClick }) => {
  return (
    <div className={styles.welcome}>
      <div className={styles.starWrap}>
        <span className={styles.orbitRing} aria-hidden="true" />
        <PoleStarIcon size={56} animated glow />
      </div>

      <h1 className={styles.title}>
        <span className={styles.brandWord}>KutupAI</span>&apos;ye Hoş Geldiniz
      </h1>
      <p className={styles.subtitle}>{CHAT_CONFIG.ui.tagline}</p>

      <span className={styles.suggestionsLabel}>Örnek Sorular</span>
      <div className={styles.suggestions}>
        {SUGGESTIONS.map((suggestion, i) => (
          <button
            key={suggestion.text}
            type="button"
            className={styles.suggestionCard}
            style={{ animationDelay: `${0.12 + i * 0.06}s` }}
            onClick={() => onSuggestionClick(suggestion.text)}
          >
            <span className={styles.suggestionIcon} aria-hidden="true">
              {suggestion.icon}
            </span>
            <span className={styles.suggestionText}>{suggestion.text}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeState;