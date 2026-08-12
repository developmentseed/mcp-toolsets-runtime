import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// One self-contained view per pass, inlined into a single HTML file in
// ../views/, where the runtime serves it as ui://raster-ops/<id>. A view is
// fetched over MCP as a resource, so it cannot reference sibling assets —
// everything has to be in the one file. Same shape as `mcp-toolset new
// --with-ui` scaffolds, because that is what a real toolset looks like.
const view = process.env.VIEW;
if (!view) throw new Error("set VIEW=<view-id> (e.g. VIEW=clip vite build)");

export default defineConfig({
  plugins: [react(), viteSingleFile()],
  build: {
    outDir: resolve(__dirname, "../views"),
    emptyOutDir: false,
    rollupOptions: { input: { [view]: resolve(__dirname, `${view}.html`) } },
  },
});
