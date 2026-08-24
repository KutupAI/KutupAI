import React from "react";
import styles from "./StatusBadge.module.css";

interface StatusBadgeProps {
  label: string;
  tone?: "positive" | "neutral";
}

/** Small pill badge used to surface boolean/status facts (e.g. imza/kaşe
 *  tespiti) inside the structured analysis card without raw JSON. */
const StatusBadge: React.FC<StatusBadgeProps> = ({ label, tone = "neutral" }) => (
  <span className={`${styles.badge} ${tone === "positive" ? styles.positive : styles.neutral}`}>
    <span className={styles.dot} aria-hidden="true" />
    {label}
  </span>
);

export default StatusBadge;
