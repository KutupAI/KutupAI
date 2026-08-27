import React, { useEffect, useState } from "react";
import Navbar from "../Navbar/Navbar";
import ChatSidebar from "../ChatSidebar/ChatSidebar";
import ChatWindow from "../ChatWindow/ChatWindow";
import MessageInput from "../MessageInput/MessageInput";
import { useChatHistory } from "../../hooks/useChatHistory";
import styles from "./KutupAIChat.module.css";

const MOBILE_BREAKPOINT = 768;

/**
 * KutupAIChat — nokta-i başlangıç: Navbar (menü aç/kapa) + ChatSidebar
 * (sohbet geçmişi) + ChatWindow (mesajlar) + MessageInput.
 *
 * Sidebar varsayılan açık/kapalı durumu ekran genişliğine göre belirlenir
 * (masaüstünde açık, mobilde kapalı) -- sadece ilk yüklemede, sonrasında
 * tamamen kullanıcı kontrolünde (Navbar'daki menü düğmesi).
 */
const KutupAIChat: React.FC = () => {
  const {
    currentChat,
    chats,
    activeChatId,
    isLoading,
    error,
    startNewChat,
    selectChat,
    deleteChat,
    sendMessage,
  } = useChatHistory();

  const [suggestionText, setSuggestionText] = useState<string | undefined>(undefined);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(
    () => typeof window !== "undefined" && window.innerWidth > MOBILE_BREAKPOINT
  );

  // Masaüstü <-> mobil arası geçişte (pencere yeniden boyutlandırma)
  // sidebar durumunu makul bir varsayılana getirir; kullanıcı sonradan
  // yine istediği gibi açıp kapatabilir.
  useEffect(() => {
    const handleResize = () => {
      setIsSidebarOpen(window.innerWidth > MOBILE_BREAKPOINT);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const handleSuggestionClick = (text: string) => {
    setSuggestionText(text);
  };

  const handleSend = (text: string, file: Parameters<typeof sendMessage>[1]) => {
    setSuggestionText(undefined);
    sendMessage(text, file);
  };

  const handleSelectChat = (id: string) => {
    selectChat(id);
    if (window.innerWidth <= MOBILE_BREAKPOINT) setIsSidebarOpen(false);
  };

  const handleNewChat = () => {
    startNewChat();
    if (window.innerWidth <= MOBILE_BREAKPOINT) setIsSidebarOpen(false);
  };

  return (
    <div className={styles.shell}>
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        isOpen={isSidebarOpen}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={deleteChat}
        onClose={() => setIsSidebarOpen(false)}
      />

      <div className={styles.appShell}>
        <Navbar
          onToggleSidebar={() => setIsSidebarOpen((v) => !v)}
          isSidebarOpen={isSidebarOpen}
        />

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
    </div>
  );
};

export default KutupAIChat;