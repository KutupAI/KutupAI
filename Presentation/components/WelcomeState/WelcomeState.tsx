import React from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import { CHAT_CONFIG } from "../../config/chatConfig";
import styles from "./WelcomeState.module.css";

const SUGGESTIONS: string[] = [
  "Bu belgeyi özetler misin?",
  "Bu başvuru için hangi mevzuat geçerli?",
  "Bu evrağı ilgili birime nasıl yönlendiririm?",
];

interface WelcomeStateProps {
  onSuggestionClick: (text: string) => void;
}

const WelcomeState: React.FC<WelcomeStateProps> = ({ onSuggestionClick }) => {
  return (
    <div className={styles.welcome}>
      <div className={styles.starWrap}>
        <PoleStarIcon size={44} animated />
      </div>

      <h1 className={styles.title}>KutupAI&apos;ye Hoş Geldiniz</h1>
      <p className={styles.subtitle}>{CHAT_CONFIG.ui.tagline}</p>

      <div className={styles.suggestions}>
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            className={styles.suggestionChip}
            onClick={() => onSuggestionClick(suggestion)}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
};

export default WelcomeState;
