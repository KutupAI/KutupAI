import React, { useRef, useState } from "react";
import FileUpload, { SelectedFilePreview } from "../FileUpload/FileUpload";
import { fileToBase64, validateChatFile } from "../../utils/fileValidators";
import { SendIcon } from "../icons/SectionIcons";
import styles from "./MessageInput.module.css";

interface PendingFile extends SelectedFilePreview {
  base64: string;
  type: string;
}

interface MessageInputProps {
  onSend: (text: string, file: PendingFile | null) => void;
  isLoading: boolean;
  prefillText?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSend, isLoading, prefillText }) => {
  const [text, setText] = useState(prefillText ?? "");
  const [pendingFile, setPendingFile] = useState<PendingFile | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Textarea, yazılan satır sayısına göre otomatik büyür (CSS'teki
  // max-height'a kadar) -- sabit tek satırlık kutuda kaydırmaya
  // zorlamak yerine, mesaj kutusu içeriği takip eder.
  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  React.useEffect(() => {
    if (prefillText) setText(prefillText);
  }, [prefillText]);

  React.useEffect(() => {
    resizeTextarea();
  }, [text]);

  const handleFileSelect = async (file: File) => {
    const validation = validateChatFile(file);
    if (!validation.isValid) {
      setFileError(validation.errorMessage ?? null);
      return;
    }
    setFileError(null);

    const base64 = await fileToBase64(file);
    const previewUrl = file.type.startsWith("image/")
      ? URL.createObjectURL(file)
      : undefined;

    setPendingFile({
      name: file.name,
      sizeBytes: file.size,
      type: file.type,
      base64,
      previewUrl,
    });
  };

  const handleRemoveFile = () => {
    if (pendingFile?.previewUrl) URL.revokeObjectURL(pendingFile.previewUrl);
    setPendingFile(null);
    setFileError(null);
  };

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (isLoading) return;
    if (!text.trim() && !pendingFile) return;

    onSend(text, pendingFile);
    setText("");
    handleRemoveFile();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <form className={styles.inputBar} onSubmit={handleSubmit}>
      {fileError && (
        <div className={styles.fileErrorBanner} role="alert">
          {fileError}
        </div>
      )}

      <FileUpload
        selectedFile={pendingFile}
        onFileSelect={handleFileSelect}
        onFileRemove={handleRemoveFile}
        disabled={isLoading}
      />

      <div className={styles.row}>
        <textarea
          ref={textareaRef}
          className={styles.textarea}
          placeholder="Mesajınızı yazın..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={isLoading}
          aria-label="Mesajınızı yazın"
        />

        <button
          type="submit"
          className={styles.sendBtn}
          disabled={isLoading || (!text.trim() && !pendingFile)}
          aria-label="Gönder"
          title="Gönder"
        >
          <SendIcon size={17} />
        </button>
      </div>
    </form>
  );
};

export default MessageInput;