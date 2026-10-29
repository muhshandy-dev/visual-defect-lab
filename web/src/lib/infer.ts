// Inferensi langsung di browser dengan onnxruntime-web (WASM, satu thread).
// Runtime dan model dimuat hanya saat pengunjung menekan tombol, bukan saat halaman dibuka.
// Berkas .wasm runtime dibundel Vite ke dist/assets dan disajikan dari domain sendiri.

import type { InferenceSession, Tensor } from "onnxruntime-web";
import { rgbaToCHW } from "./heat.ts";

export const SIZE = 256;

type Ort = typeof import("onnxruntime-web");
let ortPromise: Promise<Ort> | null = null;
let backbone: Promise<InferenceSession> | null = null;
const heads = new Map<string, Promise<InferenceSession>>();

function base(): string {
  return import.meta.env.BASE_URL;
}

async function ort(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((m) => {
      m.env.wasm.numThreads = 1;
      m.env.wasm.proxy = false;
      return m as unknown as Ort;
    });
  }
  return ortPromise;
}

async function load(url: string): Promise<InferenceSession> {
  const m = await ort();
  return m.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
}

export function backboneSession(): Promise<InferenceSession> {
  if (!backbone) backbone = load(`${base()}models/backbone_int8.onnx`);
  return backbone;
}

export function headSession(cls: string): Promise<InferenceSession> {
  let s = heads.get(cls);
  if (!s) {
    s = load(`${base()}models/head_${cls}.onnx`);
    heads.set(cls, s);
  }
  return s;
}

export interface LiveResult {
  map: Float32Array; // [256*256] skor piksel
  score: number;
  msBackbone: number;
  msHead: number;
  msLoad: number;
}

export async function runLive(rgba: Uint8ClampedArray, cls: string): Promise<LiveResult> {
  const t0 = performance.now();
  const m = await ort();
  const [bb, hd] = await Promise.all([backboneSession(), headSession(cls)]);
  const t1 = performance.now();
  const input = new m.Tensor("float32", rgbaToCHW(rgba, SIZE), [1, 3, SIZE, SIZE]);
  const feats = (await bb.run({ image: input })).patches as Tensor;
  const t2 = performance.now();
  const out = await hd.run({ patches: feats });
  const t3 = performance.now();
  const map = (out.map as Tensor).data as Float32Array;
  let score = -Infinity;
  for (let i = 0; i < map.length; i++) if (map[i] > score) score = map[i];
  return { map, score, msLoad: t1 - t0, msBackbone: t2 - t1, msHead: t3 - t2 };
}

/** Gambar citra ke kanvas 256 x 256 lalu ambil RGBA. Untuk sampel, berkas masukan sudah
 * berukuran 256 x 256 (PNG tanpa kompresi lossy) sehingga piksel sama persis dengan Python. */
export async function imageToRGBA(src: string | Blob): Promise<Uint8ClampedArray> {
  const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
  const bmp = await createImageBitmap(blob, {
    resizeWidth: SIZE,
    resizeHeight: SIZE,
    resizeQuality: "high",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const c = new OffscreenCanvas(SIZE, SIZE);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("kanvas tidak tersedia");
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, SIZE, SIZE).data;
}
