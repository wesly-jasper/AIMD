import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is reached through the dev server's proxy, so the browser only ever
// talks to one origin and the backend's CORS list does not need this port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: process.env.AIMD_API ?? "http://127.0.0.1:8000", changeOrigin: false },
    },
  },
});
