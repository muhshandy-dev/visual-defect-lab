import type { ClassSamples, Summary } from "./types.ts";

const cache = new Map<string, Promise<unknown>>();

export function url(path: string): string {
  return `${import.meta.env.BASE_URL}${path}`;
}

function getJSON<T>(path: string): Promise<T> {
  let p = cache.get(path);
  if (!p) {
    p = fetch(url(path)).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    });
    cache.set(path, p);
  }
  return p as Promise<T>;
}

export const loadSummary = () => getJSON<Summary>("data/summary.json");
export const loadSamples = (cls: string) => getJSON<ClassSamples>(`data/samples/${cls}.json`);

const gray = new Map<string, Promise<Uint8Array>>();

/** Baca PNG abu-abu 256 x 256 (peta skor terkuantisasi atau masker) menjadi Uint8Array. */
export function loadGray(path: string): Promise<Uint8Array> {
  let p = gray.get(path);
  if (!p) {
    p = (async () => {
      const blob = await (await fetch(url(path))).blob();
      const bmp = await createImageBitmap(blob, {
        colorSpaceConversion: "none",
        premultiplyAlpha: "none",
      });
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("kanvas tidak tersedia");
      ctx.drawImage(bmp, 0, 0);
      const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      bmp.close();
      const out = new Uint8Array(rgba.length / 4);
      for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4];
      return out;
    })();
    gray.set(path, p);
  }
  return p;
}
