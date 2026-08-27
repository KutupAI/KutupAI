import React from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import styles from "./Navbar.module.css";

interface NavbarProps {
  onToggleSidebar: () => void;
  isSidebarOpen: boolean;
}

const Navbar: React.FC<NavbarProps> = ({ onToggleSidebar, isSidebarOpen }) => {
  return (
    <header className={styles.navbar} role="banner">
      <button
        type="button"
        className={styles.sidebarToggle}
        onClick={onToggleSidebar}
        aria-label={isSidebarOpen ? "Kenar çubuğunu kapat" : "Sohbet geçmişini aç"}
        aria-expanded={isSidebarOpen}
      >
        {isSidebarOpen ? (
          // Sidebar açıkken: sol paneli "kapatma" fikrini taşıyan ikon.
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <rect x="3" y="4.5" width="18" height="15" rx="2.5" stroke="currentColor" strokeWidth="1.7" />
            <path d="M9.5 4.5v15" stroke="currentColor" strokeWidth="1.7" />
            <path d="M13.5 9.5 16 12l-2.5 2.5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          // Sidebar kapalıyken: standart hamburger (aç).
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <rect x="3" y="5" width="18" height="2" rx="1" fill="currentColor" />
            <rect x="3" y="11" width="18" height="2" rx="1" fill="currentColor" />
            <rect x="3" y="17" width="18" height="2" rx="1" fill="currentColor" />
          </svg>
        )}
      </button>

      <div className={styles.brand}>
        <PoleStarIcon size={26} className={styles.brandIcon} />
        <span className={styles.brandName}>KutupAI</span>
      </div>

      {/* Sağ tarafta kasıtlı olarak boş bırakıldı: hesap/oturum açma bu
          yüzeyde gösterilmiyor (bkz. ChatSidebar.tsx üst notu). */}
      <div className={styles.spacer} aria-hidden="true" />
    </header>
  );
};

export default Navbar;