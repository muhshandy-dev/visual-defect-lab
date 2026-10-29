import assert from "node:assert/strict";
import { test } from "node:test";
import {
  aboveThreshold, decodeRatio, edgeMask, encodeRatio, heatColor, iou, quantizeMap, rgbaToCHW, verdict,
} from "../src/lib/heat.ts";

test("encode: ambang tepat di 127,5 dan dibatasi 2x", () => {
  assert.equal(encodeRatio(0, 1), 0);
  assert.equal(encodeRatio(1, 1), 128); // Math.round(127.5) = 128, sama dengan floor(x + 0,5) di Python
  assert.equal(encodeRatio(0.999, 1), 127);
  assert.equal(encodeRatio(5, 1), 255);
  assert.equal(encodeRatio(-1, 1), 0);
  assert.ok(Math.abs(decodeRatio(128) - 1) < 0.01);
});

test("piksel di atas ambang = v >= 128, konsisten dengan keputusan skor > ambang", () => {
  const thr = 2.5;
  const map = new Float32Array([2.4, 2.49, 2.5, 2.6, 7]);
  const q = quantizeMap(map, thr);
  assert.deepEqual([...aboveThreshold(q)], [0, 0, 1, 1, 1]);
  assert.equal(verdict(2.6, thr), "tolak");
  assert.equal(verdict(2.5, thr), "lolos");
});

test("warna: transparan jauh di bawah ambang, buram di atas ambang", () => {
  assert.equal(heatColor(0.3)[3], 0);
  assert.ok(heatColor(1)[3] > 0.7);
  assert.ok(heatColor(2)[3] >= heatColor(1.2)[3]);
});

test("tepi masker dan IoU", () => {
  const w = 5;
  const m = new Uint8Array(25);
  for (let y = 1; y < 4; y++) for (let x = 1; x < 4; x++) m[y * w + x] = 1;
  const e = edgeMask(m, w, w);
  assert.equal(e[2 * w + 2], 0); // pusat bukan tepi
  assert.equal([...e].reduce((a, b) => a + b, 0), 8);
  const n = new Uint8Array(25);
  n[2 * w + 2] = 1;
  assert.equal(iou(m, n), 1 / 9);
  assert.equal(iou(new Uint8Array(4), new Uint8Array(4)), 1);
});

test("tensor masukan CHW = uint8 / 255 seperti to_tensor Python", () => {
  const rgba = new Uint8ClampedArray([255, 0, 51, 255, 0, 255, 102, 255, 10, 20, 30, 255, 1, 2, 3, 255]);
  const t = rgbaToCHW(rgba, 2);
  assert.equal(t.length, 12);
  assert.equal(t[0], 1);
  assert.equal(t[4], Math.fround(0));
  assert.equal(t[8], Math.fround(51 / 255));
  assert.equal(t[5], Math.fround(255 / 255));
});
