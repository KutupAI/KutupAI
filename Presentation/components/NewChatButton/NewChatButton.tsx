import React from "react";
import styles from "./NewChatButton.module.css";

interface NewChatButtonProps {
  onClick: () => void;
}

const NewChatButton: React.FC<NewChatButtonProps> = ({ onClick }) => {
  return (
    <button
      type="button"
      className={`btn ${styles.newChatBtn}`}
      onClick={onClick}
      aria-label="Yeni sohbet başlat"
    >
      <span className={styles.plus} aria-hidden="true">
        +
      </span>
      <span className={styles.label}>Yeni Sohbet</span>
    </button>
  );
};

export default NewChatButton;
