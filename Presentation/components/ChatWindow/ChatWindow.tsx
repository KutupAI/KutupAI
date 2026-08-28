import React, { useEffect, useRef, useState } from "react";
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
  isAdmin: boolean;
}

const STAGE_INTERVAL_MS = 4500;

const ThinkingIndicator: React.FC = () => {
  const stages = CHAT_CONFIG.ui.thinkingStages;
  const [stageIndex, setStageIndex] = useState(0);
  const [elapsedSec, setElapsedSec] = useState(0);

  useEffect(() => {
    const stageTimer = window.setInterval(() => {
      setStageIndex((i) => (i + 1) % stages.length);
    }, STAGE_INTERVAL_MS);

    const clockTimer = window.setInterval(() => {
      setElapsedSec((s) => s + 1);
    }, 1000);

    return () => {
      window.clearInterval(stageTimer);
      window.clearInterval(clockTimer);
    };
  }, [stages.length]);

  const progress = ((stageIndex + 1) / stages.length) * 100;
  const mm = String(Math.floor(elapsedSec / 60)).padStart(2, "0");
  const ss = String(elapsedSec % 60).padStart(2, "0");

  return (
    <div className={styles.thinkingRow} role="status" aria-live="polite">
      <div className={styles.thinkingAvatar}>
        <PoleStarIcon size={16} animated glow />
      </div>
      <div className={styles.thinkingBubble}>
        <div className={styles.thinkingTop}>
          <span className={styles.thinkingLabel} key={stageIndex}>
            {stages[stageIndex]}
          </span>
          <span className={styles.thinkingTime}>{mm}:{ss}</span>
        </div>
        <div className={styles.progressTrack} aria-hidden="true">
          <div
            className={styles.progressFill}
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className={styles.thinkingHint}>
          <span className={styles.dots} aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>{CHAT_CONFIG.ui.thinkingLabel}</span>
        </div>
      </div>
    </div>
  );
};

const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  error,
  onSuggestionClick,
  isAdmin,
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
            <ChatMessage key={message.id} message={message} isAdmin={isAdmin} />
          ))}

          {isLoading && <ThinkingIndicator />}

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
