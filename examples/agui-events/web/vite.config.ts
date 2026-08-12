import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `python -m service` serves the agent on 8765. Proxying to it means the browser
// only ever talks to one origin, so this example needs no CORS — which belongs
// to `mcp_agent_api.app` anyway, not to the router.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
