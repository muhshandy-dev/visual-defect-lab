// Logika peta skor yang murni (tanpa DOM), diuji di tests/heat.test.ts.
//
// Peta skor disimpan sebagai PNG abu-abu 8 bit dengan nilai v = round(min(skor/ambang, 2) * 127.5).
// Jadi v = 127,5 tepat di ambang operasi; v >= 128 berarti piksel itu melewati ambang.

export const ENCODE_MAX_RATIO = 2;

export function encodeRatio(score: number, threshold: number): number {
  const r = Math.min(Math.max(score / threshold, 0), ENCODE_MAX_RATIO);
  return Math.round((r / ENCODE_MAX_RATIO) * 255);
}

export function decodeRatio(v: number): number {
  return (v / 255) * ENCODE_MAX_RATIO;
}

/** Ubah peta skor float (hasil inferensi langsung) ke rasio terhadap ambang, terkuantisasi
 * sama seperti PNG precomputed, supaya tampilan live dan precomputed memakai satu jalur. */
export function quantizeMap(map: Float32Array, threshold: number): Uint8Array {
  const out = new Uint8Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = encodeRatio(map[i], threshold);
  return out;
}

export function maxOf(a: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  return m;
}

// Skala amber berurutan: transparan di bawah 0,55 x ambang, oranye tua menuju ambang,
// amber terang di ambang, krem pucat untuk skor jauh di atas ambang.
const STOPS: Array<[number, [number, number, number], number]> = [
  [0.55, [120, 53, 15], 0],
  [0.8, [180, 83, 9], 0.45],
  [1.0, [245, 158, 11], 0.72],
  [1.4, [252, 211, 77], 0.85],
  [2.0, [254, 243, 199], 0.92],
];

export function heatColor(ratio: number): [number, number, number, number] {
  if (ratio <= STOPS[0][0]) return [0, 0, 0, 0];
  for (let i = 1; i < STOPS.length; i++) {
    const [r1, c1, a1] = STOPS[i];
    if (ratio <= r1) {
      const [r0, c0, a0] = STOPS[i - 1];
      const t = (ratio - r0) / (r1 - r0);
      return [
        Math.round(c0[0] + t * (c1[0] - c0[0])),
        Math.round(c0[1] + t * (c1[1] - c0[1])),
        Math.round(c0[2] + t * (c1[2] - c0[2])),
        a0 + t * (a1 - a0),
      ];
    }
  }
  const last = STOPS[STOPS.length - 1];
  return [last[1][0], last[1][1], last[1][2], last[2]];
}

/** RGBA untuk lapisan heatmap dari peta terkuantisasi (panjang w*h). */
export function heatRGBA(q: Uint8Array): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(q.length * 4);
  const lut = new Array(256).fill(0).map((_, v) => heatColor(decodeRatio(v)));
  for (let i = 0; i < q.length; i++) {
    const c = lut[q[i]];
    out[i * 4] = c[0];
    out[i * 4 + 1] = c[1];
    out[i * 4 + 2] = c[2];
    out[i * 4 + 3] = Math.round(c[3] * 255);
  }
  return out;
}

/** Piksel tepi dari masker biner (4 tetangga). Dipakai untuk garis area terdeteksi dan
 * garis cacat sebenarnya. */
export function edgeMask(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      if (
        x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
        !mask[i - 1] || !mask[i + 1] || !mask[i - w] || !mask[i + w]
      ) {
        out[i] = 1;
      }
    }
  }
  return out;
}

export function aboveThreshold(q: Uint8Array): Uint8Array {
  const out = new Uint8Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] >= 128 ? 1 : 0;
  return out;
}

export function iou(a: Uint8Array, b: Uint8Array): number {
  let inter = 0;
  let uni = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++;
    if (a[i] || b[i]) uni++;
  }
  return uni === 0 ? 1 : inter / uni;
}

export type Verdict = "tolak" | "lolos";

export function verdict(score: number, threshold: number): Verdict {
  return score > threshold ? "tolak" : "lolos";
}

/** Siapkan tensor masukan [1,3,S,S] float 0..1 dari RGBA kanvas, sama seperti to_tensor
 * di Python (uint8 / 255). */
export function rgbaToCHW(rgba: Uint8ClampedArray | Uint8Array, size: number): Float32Array {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4] / 255;
    out[n + i] = rgba[i * 4 + 1] / 255;
    out[2 * n + i] = rgba[i * 4 + 2] / 255;
  }
  return out;
}
