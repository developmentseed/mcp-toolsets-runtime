// The ui://raster-ops/clip view: the area clip_raster clipped to, drawn.
//
// It runs in a sandboxed iframe with an opaque origin and reaches its host over
// the MCP Apps `ui/*` JSON-RPC-over-postMessage protocol — the one Claude,
// ChatGPT, Goose and VS Code speak. `@developmentseed/mcp-view` is that bridge,
// two functions wide: `onData` for the tool's structuredContent, `sendMessage`
// to put a turn back into the conversation.
//
// Worth being clear about what arrives here. `geometry` is a 2000-vertex
// polygon that was never in the model's context: the model called clip_raster
// with a dataset id and nothing else, session state filled `aoi` from what an
// earlier tool published, and the reply was captured back out before the
// transcript was written. The host reassembles it for this view alone.
import { configure, onData, sendMessage } from "@developmentseed/mcp-view";
import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";

import "./clip.css";

configure({ name: "clip-view", version: "1.0.0" });

type Clip = {
  message?: string;
  dataset?: string;
  vertices?: number;
  bounds?: [number, number, number, number];
  geometry?: {
    features?: { geometry?: { type?: string; coordinates?: number[][][] } }[];
  };
};

/** Every ring in the collection, projected into the SVG's box.
 *
 * Equirectangular and unapologetic: at catchment scale the distortion is
 * invisible, and a projection library would be most of the bundle.
 *
 * Each axis is fitted independently rather than to a common span. That is the
 * wrong choice for a map and the right one here, because the session-state
 * example's area of interest is a synthetic ring — 2000 vertices generated to
 * cost tokens, a degree wide and three thousandths of a degree tall. Fitted to
 * a square it is one flat line. Give this view a real catchment and both axes
 * scale together anyway.
 */
function paths(clip: Clip, size: number, pad: number): string[] {
  const [west, south, east, north] = clip.bounds ?? [0, 0, 1, 1];
  const width = east - west || 1;
  const height = north - south || 1;
  const x = (lon: number) => pad + ((lon - west) / width) * (size - 2 * pad);
  const y = (lat: number) => size - pad - ((lat - south) / height) * (size - 2 * pad);

  return (clip.geometry?.features ?? []).flatMap((feature) => {
    const shape = feature.geometry;
    const rings =
      shape?.type === "MultiPolygon"
        ? (shape.coordinates as unknown as number[][][][]).flat()
        : ((shape?.coordinates ?? []) as number[][][]);
    return rings.map(
      (ring) =>
        `M${ring.map((point) => `${x(point[0])},${y(point[1])}`).join("L")}Z`,
    );
  });
}

function Clipped() {
  const [clip, setClip] = useState<Clip | null>(null);

  useEffect(() => onData<Clip>(setClip), []);

  if (!clip) return <p className="dim">waiting for the tool result…</p>;

  const [west, south, east, north] = clip.bounds ?? [0, 0, 0, 0];
  const shapes = paths(clip, 200, 8);

  return (
    <>
      <h1>{clip.dataset ?? "raster"}</h1>
      <div className="split">
        <svg viewBox="0 0 200 200" role="img" aria-label="clipped area">
          <rect x="0" y="0" width="200" height="200" className="sea" />
          {shapes.map((path, index) => (
            <path key={index} d={path} className="land" />
          ))}
        </svg>
        <dl>
          <dt>vertices</dt>
          <dd>{clip.vertices?.toLocaleString() ?? "—"}</dd>
          <dt>west, south</dt>
          <dd>
            {west.toFixed(3)}, {south.toFixed(3)}
          </dd>
          <dt>east, north</dt>
          <dd>
            {east.toFixed(3)}, {north.toFixed(3)}
          </dd>
        </dl>
      </div>
      {/* The other half of the bridge: a view can start the next turn. The
          geometry stays here — what goes back is a sentence. */}
      <button onClick={() => sendMessage(`Summarise ${clip.dataset} over this area.`)}>
        Summarise this area
      </button>
    </>
  );
}

createRoot(document.getElementById("root")!).render(<Clipped />);
