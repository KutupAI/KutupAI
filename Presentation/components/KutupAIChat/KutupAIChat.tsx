import React, { useState } from "react";
import Navbar from "../Navbar/Navbar";
import ChatWindow from "../ChatWindow/ChatWindow";
import MessageInput from "../MessageInput/MessageInput";
import { useChat } from "../../hooks/useChat";
import styles from "./KutupAIChat.module.css";

/**
 * KutupAIChat — نقطة الدخول الرئيسية لواجهة المحادثة.
 * يجمع بين: Navbar (+ زر Yeni Sohbet) / ChatWindow (رسائل + Welcome State)
 * / MessageInput (كتابة + رفع ملف + إرسال)، ويدير الحالة عبر useChat.
 *
 * الاستخدام داخل المشروع الحالي، مثال ضمن صفحة موجودة:
 *   import KutupAIChat from "../../components/KutupAIChat/KutupAIChat";
 *   <KutupAIChat />
 */
const KutupAIChat: React.FC = () => {
  const { currentChat, isLoading, error, startNewChat, sendMessage } = useChat();
  const [suggestionText, setSuggestionText] = useState<string | undefined>(undefined);

  const handleSuggestionClick = (text: string) => {
    setSuggestionText(text);
  };

  const handleSend = (text: string, file: Parameters<typeof sendMessage>[1]) => {
    setSuggestionText(undefined);
    sendMessage(text, file);
  };

  return (
    <div className={styles.appShell}>
      <Navbar onNewChat={startNewChat} />

      <ChatWindow
        messages={currentChat.messages}
        isLoading={isLoading}
        error={error}
        onSuggestionClick={handleSuggestionClick}
      />

      <div className={styles.inputArea}>
        <div className={styles.inputInner}>
          <MessageInput onSend={handleSend} isLoading={isLoading} prefillText={suggestionText} />
        </div>
      </div>
    </div>
  );
};

export default KutupAIChat;
