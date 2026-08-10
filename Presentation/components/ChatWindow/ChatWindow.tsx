import React, { useEffect, useRef } from "react";
import ChatMessage from "../ChatMessage/ChatMessage";
import WelcomeState from "../WelcomeState/WelcomeState";
import PoleStarIcon from "../icons/PoleStarIcon";
import { ChatMessageModel } from "../../models/ChatMessage";
import { CHAT_CONFIG } from "../../config/chatConfig";
import styles from "./ChatWindow.module.css";

interface ChatWindowProps {
  messages: ChatMessageModel[];
  isLoading: boolean;
  error: string | null;
  onSuggestionClick: (text: string) => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  error,
  onSuggestionClick,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, isLoading]);

  const isEmpty = messages.length === 0;

  return (
    <main className={styles.window} role="log" aria-live="polite">
      {isEmpty ? (
        <WelcomeState onSuggestionClick={onSuggestionClick} />
      ) : (
        <div className={styles.messageList}>
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isLoading && (
            <div className={styles.thinkingRow}>
              <div className={styles.thinkingAvatar}>
                <PoleStarIcon size={16} animated />
              </div>
              <div className={styles.thinkingBubble}>
                <span>{CHAT_CONFIG.ui.thinkingLabel}</span>
                <span className={styles.dots} aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            </div>
          )}

          {error && (
            <div className={styles.errorBanner} role="alert">
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      )}
    </main>
  );
};

export default ChatWindow;
