import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  envDir: __dirname,
  resolve: {
    alias: {
      "@presentation": path.resolve(__dirname, "../../Presentation"),
    },
  },
  server: {
    port: 5173,
    open: true,
    fs: {
      allow: [path.resolve(__dirname, "../..")],
    },
  },
});
