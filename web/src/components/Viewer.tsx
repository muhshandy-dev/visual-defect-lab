import { useCallback, useEffect, useRef, useState } from "react";
import { aboveThreshold, edgeMask, heatRGBA } from "../lib/heat.ts";

const N = 256;
const MAX_SCALE = 6;

export interface Layers {
  heat: boolean;
  detected: boolean;
  truth: boolean;
}

interface Props {
  img: string;
  w: number;
  h: number;
  alt: string;
  map: Uint8Array | null;
  mask: Uint8Array | null;
  layers: Layers;
}

interface View {
  s: number;
  x: number;
  y: number;
}

function cssColor(name: string, fallback: [number, number, number]): [number, number, number] {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const m = /^#([0-9a-f]{6})$/i.exec(v);
  if (!m) return fallback;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function Viewer({ img, w, h, alt, map, mask, layers }: Props) {
  const stage = useRef<HTMLDivElement>(null);
  const heat = useRef<HTMLCanvasElement>(null);
  const lines = useRef<HTMLCanvasElement>(null);
  const [split, setSplit] = useState(50);
  const [view, setView] = useState<View>({ s: 1, x: 0, y: 0 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{ dist: number; view: View; cx: number; cy: number } | null>(null);
  const dragSplit = useRef(false);
  const [theme, setTheme] = useState(0);

  // Gambar ulang garis saat tema berganti (warna garis cacat sebenarnya mengikuti tema).
  useEffect(() => {
    const obs = new MutationObserver(() => setTheme((t) => t + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    setView({ s: 1, x: 0, y: 0 });
  }, [img]);

  useEffect(() => {
    const c = heat.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, N, N);
    if (map && layers.heat) ctx.putImageData(new ImageData(heatRGBA(map), N, N), 0, 0);
  }, [map, layers.heat]);

  useEffect(() => {
    const c = lines.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const data = new Uint8ClampedArray(N * N * 4);
    const paint = (edges: Uint8Array, rgb: [number, number, number]) => {
      for (let i = 0; i < edges.length; i++) {
        if (!edges[i]) continue;
        data[i * 4] = rgb[0];
        data[i * 4 + 1] = rgb[1];
        data[i * 4 + 2] = rgb[2];
        data[i * 4 + 3] = 255;
      }
    };
    if (map && layers.detected) paint(edgeMask(aboveThreshold(map), N, N), cssColor("--detect", [251, 191, 36]));
    if (mask && layers.truth) paint(edgeMask(mask, N, N), cssColor("--truth", [240, 240, 240]));
    ctx.putImageData(new ImageData(data, N, N), 0, 0);
  }, [map, mask, layers.detected, layers.truth, theme]);

  const clamp = useCallback((v: View): View => {
    const el = stage.current;
    if (!el) return v;
    const W = el.clientWidth;
    const H = el.clientHeight;
    const s = Math.min(Math.max(v.s, 1), MAX_SCALE);
    return {
      s,
      x: Math.min(0, Math.max(W - s * W, v.x)),
      y: Math.min(0, Math.max(H - s * H, v.y)),
    };
  }, []);

  const zoomAt = useCallback(
    (factor: number, cx?: number, cy?: number) => {
      const el = stage.current;
      if (!el) return;
      const px = cx ?? el.clientWidth / 2;
      const py = cy ?? el.clientHeight / 2;
      setView((v) => {
        const s = Math.min(Math.max(v.s * factor, 1), MAX_SCALE);
        return clamp({ s, x: px - ((px - v.x) * s) / v.s, y: py - ((py - v.y) * s) / v.s });
      });
    },
    [clamp],
  );

  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomAt]);

  const local = (e: React.PointerEvent) => {
    const r = stage.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const onDown = (e: React.PointerEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest(".split-handle")) {
      dragSplit.current = true;
    }
    stage.current!.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, local(e));
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()];
      gesture.current = {
        dist: Math.hypot(a.x - b.x, a.y - b.y),
        view,
        cx: (a.x + b.x) / 2,
        cy: (a.y + b.y) / 2,
      };
      dragSplit.current = false;
    }
  };

  const onMove = (e: React.PointerEvent) => {
    const prev = pointers.current.get(e.pointerId);
    if (!prev) return;
    const p = local(e);
    pointers.current.set(e.pointerId, p);
    if (dragSplit.current) {
      const el = stage.current!;
      // Posisi pembanding dalam koordinat gambar (memperhitungkan zoom).
      const xImg = (p.x - view.x) / view.s;
      setSplit(Math.min(100, Math.max(0, (xImg / el.clientWidth) * 100)));
      return;
    }
    if (pointers.current.size === 2 && gesture.current) {
      const [a, b] = [...pointers.current.values()];
      const g = gesture.current;
      const s = Math.min(Math.max((g.view.s * Math.hypot(a.x - b.x, a.y - b.y)) / g.dist, 1), MAX_SCALE);
      setView(clamp({ s, x: g.cx - ((g.cx - g.view.x) * s) / g.view.s, y: g.cy - ((g.cy - g.view.y) * s) / g.view.s }));
      return;
    }
    if (pointers.current.size === 1 && view.s > 1) {
      setView((v) => clamp({ s: v.s, x: v.x + p.x - prev.x, y: v.y + p.y - prev.y }));
    }
  };

  const onUp = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    if (pointers.current.size < 2) gesture.current = null;
    if (pointers.current.size === 0) dragSplit.current = false;
  };

  const zoomPct = Math.round(view.s * 100);

  return (
    <div className="viewer">
      <div
        ref={stage}
        className={`stage${view.s > 1 ? " is-zoomed" : ""}`}
        style={{ aspectRatio: `${w} / ${h}`, width: `min(100%, ${((68 * w) / h).toFixed(2)}vh)` }}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
        onDoubleClick={(e) => {
          const r = stage.current!.getBoundingClientRect();
          if (view.s > 1) setView({ s: 1, x: 0, y: 0 });
          else zoomAt(2.5, e.clientX - r.left, e.clientY - r.top);
        }}
      >
        <div
          className="zoomable"
          style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}
        >
          <img className="layer" src={img} alt={alt} draggable={false} />
          <div className="layer after" style={{ clipPath: `inset(0 0 0 ${split}%)` }}>
            <img className="layer dim" src={img} alt="" draggable={false} />
            <canvas ref={heat} className="layer heat" width={N} height={N} />
          </div>
          <canvas ref={lines} className="layer lines" width={N} height={N} />
        </div>
        <div className="split-line" style={{ left: `calc(${view.x}px + ${split * view.s}%)` }}>
          <span className="split-handle" aria-hidden="true">
            <svg viewBox="0 0 16 16" width="14" height="14">
              <path d="M6 3 2 8l4 5M10 3l4 5-4 5" fill="none" stroke="currentColor" strokeWidth="1.6" />
            </svg>
          </span>
        </div>
        <span className="stage-tag left">Foto asli</span>
        <span className="stage-tag right">Peta skor</span>
      </div>
      <div className="viewer-tools">
        <label className="split-range">
          <span>Pembanding</span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(split)}
            onChange={(e) => setSplit(Number(e.target.value))}
            aria-label="Geser batas antara foto asli dan peta skor"
          />
        </label>
        <div className="zoom-tools" role="group" aria-label="Perbesaran">
          <button type="button" onClick={() => zoomAt(1 / 1.5)} disabled={view.s <= 1} aria-label="Perkecil">
            −
          </button>
          <span className="mono" aria-live="polite">{zoomPct}%</span>
          <button type="button" onClick={() => zoomAt(1.5)} disabled={view.s >= MAX_SCALE} aria-label="Perbesar">
            +
          </button>
          <button type="button" onClick={() => setView({ s: 1, x: 0, y: 0 })} disabled={view.s === 1}>
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
