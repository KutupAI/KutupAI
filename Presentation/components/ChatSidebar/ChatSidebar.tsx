import React, { useMemo, useState } from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import { PlusIcon, TrashIcon, CheckIcon } from "../icons/SectionIcons";
import { ChatModel } from "../../models/Chat";
import { groupChatsByRecency } from "../../utils/formatters";
import { CHAT_CONFIG } from "../../config/chatConfig";
import styles from "./ChatSidebar.module.css";

interface ChatSidebarProps {
  chats: ChatModel[];
  activeChatId: string | null;
  isOpen: boolean;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  onClose: () => void;
}

/**
 * Sohbet geçmişi kenar çubuğu. Bilinçli olarak İÇERMEDİĞİ şeyler: hesap /
 * oturum açma / kullanıcı adı bölümü (ürün henüz kimlik doğrulama akışını
 * bu yüzeyde göstermiyor -- bkz. proje talebi). Sadece: marka, yeni sohbet,
 * arama, ve zaman bazlı gruplanmış geçmiş listesi.
 */
const ChatSidebar: React.FC<ChatSidebarProps> = ({
  chats,
  activeChatId,
  isOpen,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  onClose,
}) => {
  const [query, setQuery] = useState("");
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return chats;
    return chats.filter((c) => c.title.toLowerCase().includes(q));
  }, [chats, query]);

  const groups = useMemo(() => groupChatsByRecency(filtered), [filtered]);

  const handleDeleteClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (pendingDeleteId === id) {
      onDeleteChat(id);
      setPendingDeleteId(null);
    } else {
      setPendingDeleteId(id);
    }
  };

  return (
    <>
      {isOpen && <div className={styles.scrim} onClick={onClose} aria-hidden="true" />}

      <aside className={`${styles.sidebar} ${isOpen ? styles.sidebarOpen : ""}`} aria-label="Sohbet geçmişi">
        <div className={styles.brand}>
          <PoleStarIcon size={22} />
          <span className={styles.brandName}>{CHAT_CONFIG.ui.productName}</span>
        </div>

        <button type="button" className={styles.newChatBtn} onClick={onNewChat}>
          <PlusIcon size={16} />
          Yeni Sohbet
        </button>

        <div className={styles.searchWrap}>
          <svg className={styles.searchIcon} width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M20 20L16.5 16.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <input
            type="text"
            className={styles.searchInput}
            placeholder="Sohbetlerde ara..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            aria-label="Sohbetlerde ara"
          />
          {query && (
            <button
              type="button"
              className={styles.searchClear}
              onClick={() => setQuery("")}
              aria-label="Aramayı temizle"
            >
              ×
            </button>
          )}
        </div>

        <nav className={styles.history}>
          {groups.length === 0 ? (
            <div className={styles.emptyState}>
              <PoleStarIcon size={26} />
              <p>{query ? "Eşleşen sohbet yok." : "Henüz sohbet geçmişi yok."}</p>
            </div>
          ) : (
            groups.map((group) => (
              <div key={group.label} className={styles.group}>
                <h2 className={styles.groupLabel}>{group.label}</h2>
                <ul className={styles.list}>
                  {group.items.map((chat) => (
                    <li key={chat.id}>
                      <button
                        type="button"
                        className={`${styles.item} ${chat.id === activeChatId ? styles.itemActive : ""}`}
                        onClick={() => chat.id && onSelectChat(chat.id)}
                        onBlur={() => setPendingDeleteId(null)}
                      >
                        <span className={styles.itemTitle}>{chat.title}</span>
                        <span
                          role="button"
                          tabIndex={0}
                          className={`${styles.deleteBtn} ${pendingDeleteId === chat.id ? styles.deleteBtnConfirm : ""}`}
                          onClick={(e) => chat.id && handleDeleteClick(e, chat.id)}
                          onKeyDown={(e) => {
                            if ((e.key === "Enter" || e.key === " ") && chat.id) {
                              e.preventDefault();
                              handleDeleteClick(e as unknown as React.MouseEvent, chat.id);
                            }
                          }}
                          aria-label={pendingDeleteId === chat.id ? "Silmeyi onayla" : "Sohbeti sil"}
                          title={pendingDeleteId === chat.id ? "Silmeyi onayla" : "Sohbeti sil"}
                        >
                          {pendingDeleteId === chat.id ? <CheckIcon size={13} /> : <TrashIcon size={13} />}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))
          )}
        </nav>
      </aside>
    </>
  );
};

export default ChatSidebar;