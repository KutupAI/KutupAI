import { CHAT_CONFIG } from "../config/chatConfig";

export interface FileValidationResult {
  isValid: boolean;
  errorMessage?: string;
}

const getExtension = (fileName: string): string =>
  fileName.split(".").pop()?.toLowerCase() ?? "";


export const validateChatFile = (file: File): FileValidationResult => {
  const extension = getExtension(file.name);
  const allowed = CHAT_CONFIG.upload.allowedExtensions as readonly string[];

  if (!allowed.includes(extension)) {
    return {
      isValid: false,
      errorMessage:
        "Desteklenmeyen dosya türü. Yalnızca PDF, JPG, PNG ve TXT dosyaları yükleyebilirsiniz.",
    };
  }

  if (file.size > CHAT_CONFIG.upload.maxSizeBytes) {
    return {
      isValid: false,
      errorMessage: "Dosya boyutu çok büyük. En fazla 10MB yükleyebilirsiniz.",
    };
  }

  return { isValid: true };
};

export const isImageFile = (fileName: string): boolean =>
  ["jpg", "jpeg", "png"].includes(getExtension(fileName));

export const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1] ?? result;
      resolve(base64);
    };
    reader.onerror = () => reject(new Error("Dosya okunamadı."));
    reader.readAsDataURL(file);
  });

export const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};
