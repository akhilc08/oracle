import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": "http://localhost:8000",
      "/stream": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/knowledge": "http://localhost:8000/api/v1",
    },
  },
});
