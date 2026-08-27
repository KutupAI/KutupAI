import React from "react";


interface IconProps {
  size?: number;
}

/** Yanıt  */
export const AnswerIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M4 5.5C4 4.67 4.67 4 5.5 4h13c.83 0 1.5.67 1.5 1.5v9c0 .83-.67 1.5-1.5 1.5H9l-4 3.5V16H5.5C4.67 16 4 15.33 4 14.5v-9Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="round"
    />
    <path d="M8 8.5h8M8 11.5h5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

/** Özet (list). */
export const SummaryIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <rect x="4" y="4.5" width="16" height="15" rx="2" stroke="currentColor" strokeWidth="1.6" />
    <path d="M7.5 9h9M7.5 12.5h9M7.5 16h6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

/** Kaynak — */
export const SourceIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M12 6.5c-1.4-1.1-3.4-1.6-6-1.6v11.6c2.6 0 4.6.5 6 1.6c1.4-1.1 3.4-1.6 6-1.6V4.9c-2.6 0-4.6.5-6 1.6Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="round"
    />
    <path d="M12 6.5v11.6" stroke="currentColor" strokeWidth="1.6" />
  </svg>
);

/** Belge Bilgileri / Çıkarılan Bilgiler —*/
export const DocumentIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M7 3.5h7l4 4v12.5a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-15.5a1 1 0 0 1 1-1Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="round"
    />
    <path d="M14 3.5V8h4" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    <path d="M8.5 13h7M8.5 16.5h7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
  </svg>
);

/**
 * Pusula — WelcomeState öneri kartlarında "yönlendirme" temalı soru için.
 * "Kutup Yıldızı" (yön gösteren sabit ışık) marka hikayesiyle doğrudan
 * bağlantılı bir seçim: kullanıcıyı doğru birime "yönlendirme" fikri.
 */
export const CompassIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <circle cx="12" cy="12" r="8.25" stroke="currentColor" strokeWidth="1.6" />
    <path
      d="M14.8 9.2 13.1 13.1 9.2 14.8 10.9 10.9 14.8 9.2Z"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinejoin="round"
    />
  </svg>
);

/** Yeni Sohbet — "+" düğmesi (ChatSidebar). */
export const PlusIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

/** Sohbeti sil — çöp kutusu (ChatSidebar liste öğesi hover'ı). */
export const TrashIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M5 7h14M9.5 7V5.2c0-.66.54-1.2 1.2-1.2h2.6c.66 0 1.2.54 1.2 1.2V7M7.5 7l.6 12.1c.04.77.67 1.4 1.44 1.4h5.12c.77 0 1.4-.63 1.44-1.4L16.7 7"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** Silmeyi onayla — ikinci tıklamada çöp kutusunun yerine geçen onay işareti. */
export const CheckIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M5 12.5 9.5 17 19 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

/** Dosya ekle — ataç (MessageInput / FileUpload). */
export const PaperclipIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M17.5 8.5 9.9 16.1a3 3 0 1 1-4.24-4.24l8.13-8.13a2 2 0 1 1 2.83 2.83l-7.9 7.9a1 1 0 1 1-1.41-1.42l7.13-7.13"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

/** Seçilen dosya önizlemesi (FileUpload) — genel dosya simgesi. */
export const FileIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M7 3.5h7l4 4v12.5a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-15.5a1 1 0 0 1 1-1Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="round"
    />
    <path d="M14 3.5V8h4" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
  </svg>
);

/** Kapat / kaldır — küçük "×" (dosya önizlemesini kaldırma vb). */
export const CloseIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M6 6l12 12M18 6 6 18" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

/** Gönder — mesaj gönderme düğmesi (MessageInput). */
export const SendIcon: React.FC<IconProps> = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M4.5 12 19.5 5 13 19.5l-2-6.5-6.5-1Z"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinejoin="round"
      strokeLinecap="round"
    />
  </svg>
);