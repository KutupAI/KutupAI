import React, { useEffect, useState } from "react";
import Navbar from "../Navbar/Navbar";
import ChatSidebar from "../ChatSidebar/ChatSidebar";
import ChatWindow from "../ChatWindow/ChatWindow";
import MessageInput from "../MessageInput/MessageInput";
import { useChatHistory } from "../../hooks/useChatHistory";
import styles from "./KutupAIChat.module.css";

const MOBILE_BREAKPOINT = 768;
const ADMIN_SESSION_KEY = "kutupai-demo-admin";
const DEMO_ADMIN_EMAIL = "kutup@kutupai.local";
const DEMO_ADMIN_PASSWORD = "159753";

interface AdminLoginModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const AdminLoginModal: React.FC<AdminLoginModalProps> = ({ onClose, onSuccess }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (email.trim().toLowerCase() !== DEMO_ADMIN_EMAIL || password !== DEMO_ADMIN_PASSWORD) {
      setError("E-posta veya şifre hatalı.");
      return;
    }
    window.sessionStorage.setItem(ADMIN_SESSION_KEY, "true");
    onSuccess();
  };

  return (
    <div className={styles.adminModalLayer} role="dialog" aria-modal="true" aria-label="Yönetici Girişi">
      <form className={styles.adminModal} onSubmit={submit}>
        <div className={styles.adminModalMark} aria-hidden="true">✦</div>
        <h2>Yönetici Girişi</h2>
        <p>Belge özeti ve inceleme ayrıntılarına erişmek için giriş yapın.</p>
        <label>E-posta</label>
        <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="E-posta adresiniz" autoFocus />
        <label>Şifre</label>
        <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Şifreniz" />
        <div className={styles.adminLoginError} role="alert">{error}</div>
        <div className={styles.adminModalActions}>
          <button type="button" onClick={onClose}>Vazgeç</button>
          <button type="submit" className={styles.adminSubmit}>Giriş yap</button>
        </div>
      </form>
    </div>
  );
};

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
  const [isAdmin, setIsAdmin] = useState<boolean>(
    () => typeof window !== "undefined" && window.sessionStorage.getItem(ADMIN_SESSION_KEY) === "true"
  );
  const [isAdminLoginOpen, setIsAdminLoginOpen] = useState(false);

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
        isAdmin={isAdmin}
        onAdminAccess={() => setIsAdminLoginOpen(true)}
        onAdminLogout={() => {
          window.sessionStorage.removeItem(ADMIN_SESSION_KEY);
          setIsAdmin(false);
        }}
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
          isAdmin={isAdmin}
        />

        <div className={styles.inputArea}>
          <div className={styles.inputInner}>
            <MessageInput onSend={handleSend} isLoading={isLoading} prefillText={suggestionText} />
          </div>
        </div>
      </div>

      {isAdminLoginOpen && (
        <AdminLoginModal
          onClose={() => setIsAdminLoginOpen(false)}
          onSuccess={() => {
            setIsAdmin(true);
            setIsAdminLoginOpen(false);
          }}
        />
      )}
    </div>
  );
};

export default KutupAIChat;
