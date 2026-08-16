import React from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import { ChatMessageModel } from "../../models/ChatMessage";
import { isImageFile } from "../../utils/fileValidators";
import styles from "./ChatMessage.module.css";

interface ChatMessageProps {
  message: ChatMessageModel;
}

/**
 * محلّل Markdown خفيف جداً (بدون مكتبات خارجية جديدة): يدعم فقرات
 * وقوائم نقطية (- أو *) فقط، وهو كافٍ لعرض ردود AI المهيكلة.
 */
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

  return (
    <div
      className={`${styles.row} ${isUser ? styles.rowUser : styles.rowAssistant}`}
    >
      {!isUser && (
        <div className={styles.avatar} aria-hidden="true">
          <PoleStarIcon size={18} />
        </div>
      )}

      <div className={`${styles.bubble} ${isUser ? styles.bubbleUser : styles.bubbleAssistant}`}>
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
                📄
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
    </div>
  );
};

export default ChatMessage;
