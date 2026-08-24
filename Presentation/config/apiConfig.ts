/**
 * Base URL for the Application layer (Drogon).
 * In Vite preview, leave empty and use the `/api` proxy in vite.config.ts
 * so the browser stays same-origin.
 */
export const API_CONFIG = {
  /** Empty = relative URLs (recommended with Vite proxy). Override with VITE_APP_API_BASE_URL. */
  baseUrl:
    (typeof import.meta !== "undefined" &&
      (import.meta as ImportMeta & { env?: Record<string, string> }).env
        ?.VITE_APP_API_BASE_URL) ||
    "",
  timeoutMs: 320_000, // Orchestration can take up to ~300s
} as const;

export const resolveApiUrl = (path: string): string => {
  const base = API_CONFIG.baseUrl.replace(/\/$/, "");
  if (!base) return path;
  return `${base}${path.startsWith("/") ? path : `/${path}`}`;
};
