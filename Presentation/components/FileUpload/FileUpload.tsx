import React, { useRef } from "react";
import { CHAT_CONFIG } from "../../config/chatConfig";
import { formatFileSize, isImageFile } from "../../utils/fileValidators";
import styles from "./FileUpload.module.css";

export interface SelectedFilePreview {
  name: string;
  sizeBytes: number;
  previewUrl?: string;
}

interface FileUploadProps {
  selectedFile: SelectedFilePreview | null;
  onFileSelect: (file: File) => void;
  onFileRemove: () => void;
  disabled?: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({
  selectedFile,
  onFileSelect,
  onFileRemove,
  disabled,
}) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
    e.target.value = "";
  };

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={CHAT_CONFIG.upload.acceptAttribute}
        onChange={handleChange}
        className="visually-hidden"
        aria-label="Dosya seç"
        disabled={disabled}
      />

      {!selectedFile && (
        <button
          type="button"
          className={styles.pickBtn}
          onClick={() => inputRef.current?.click()}
          aria-label="Dosya yükle"
          disabled={disabled}
          title="Dosya yükle"
        >
          📎
        </button>
      )}

      {selectedFile && (
        <div className={styles.previewChip}>
          {isImageFile(selectedFile.name) && selectedFile.previewUrl ? (
            <img
              src={selectedFile.previewUrl}
              alt={selectedFile.name}
              className={styles.thumb}
            />
          ) : (
            <span className={styles.fileIcon} aria-hidden="true">
              📄
            </span>
          )}
          <div className={styles.fileMeta}>
            <span className={styles.fileName}>{selectedFile.name}</span>
            <span className={styles.fileSize}>
              {formatFileSize(selectedFile.sizeBytes)}
            </span>
          </div>
          <button
            type="button"
            className={styles.removeBtn}
            onClick={onFileRemove}
            aria-label="Dosyayı kaldır"
            title="Dosyayı kaldır"
          >
            ×
          </button>
        </div>
      )}
    </>
  );
};

export default FileUpload;
