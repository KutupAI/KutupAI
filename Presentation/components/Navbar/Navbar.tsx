import React from "react";
import PoleStarIcon from "../icons/PoleStarIcon";
import NewChatButton from "../NewChatButton/NewChatButton";
import styles from "./Navbar.module.css";

interface NavbarProps {
  onNewChat: () => void;
}

const Navbar: React.FC<NavbarProps> = ({ onNewChat }) => {
  return (
    <header className={styles.navbar} role="banner">
      <div className={`${styles.brand}`}>
        <PoleStarIcon size={26} className={styles.brandIcon} />
        <span className={styles.brandName}>KutupAI</span>
      </div>

      <NewChatButton onClick={onNewChat} />
    </header>
  );
};

export default Navbar;
