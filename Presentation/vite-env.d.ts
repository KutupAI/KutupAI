/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CHAT_DEMO?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
