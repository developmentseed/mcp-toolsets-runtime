import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build lands inside the Python package, because the wheel is how this
// client ships: `mcp_agent_api.ui` serves what is written here, and a
// deployment installs one thing rather than building a front end of its own.
//
// `base: "./"` so the page's asset URLs are relative and survive being mounted
// under a path prefix. The server redirects a bare prefix to its trailing-slash
// form, which is what relative URLs need to resolve.
//
// In development the dev server proxies `/api` to `python -m service`, so the
// browser sees one origin and this example needs no CORS — which belongs to
// `mcp_agent_api.app` anyway, not to the router.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../../src/mcp_agent_api/ui",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
