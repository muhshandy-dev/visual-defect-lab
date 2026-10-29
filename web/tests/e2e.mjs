// Uji e2e pada hasil build (dist/) yang disajikan dengan header _headers asli.
// 1. Lima halaman di 3 lebar layar x 2 tema: tanpa error konsol/CSP, tanpa scroll horizontal,
//    tanpa pelanggaran aksesibilitas serius (axe).
// 2. Interaksi penampil: zoom, pembanding, lapisan.
// 3. Bila model browser tersedia: inferensi langsung onnxruntime-web pada foto sampel dan
//    paritas skor dengan onnxruntime Python (scores.resnet18_int8 di data sampel).
import AxeBuilder from "@axe-core/playwright";
import { mkdirSync, readFileSync } from "node:fs";
import { chromium } from "playwright";
import { startServer } from "./serve.mjs";

const SHOT_DIR = process.env.SHOT_DIR;
if (SHOT_DIR) mkdirSync(SHOT_DIR, { recursive: true });
const summary = JSON.parse(readFileSync(new URL("../dist/data/summary.json", import.meta.url)));
const server = await startServer(0);
const base = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const failures = [];
let checks = 0;
const check = (ok, msg) => {
  checks++;
  if (!ok) failures.push(msg);
};

const PAGES = ["inspeksi", "galeri", "kinerja", "kesalahan", "metode"];
const VIEWPORTS = [
  [1440, 900],
  [768, 1024],
  [375, 812],
];

for (const [w, h] of VIEWPORTS) {
  for (const scheme of ["dark", "light"]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: scheme });
    const page = await ctx.newPage();
    const errors = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(String(e)));
    for (const p of PAGES) {
      await page.goto(`${base}#/${p}`);
      await page.waitForLoadState("networkidle");
      await page.waitForTimeout(300);
      const tag = `${p} ${w}px ${scheme}`;
      check(errors.length === 0, `${tag}: error konsol ${errors.join(" | ")}`);
      errors.length = 0;
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      check(overflow <= 1, `${tag}: scroll horizontal ${overflow}px`);
      const theme = await page.evaluate(() => document.documentElement.dataset.theme);
      check(theme === scheme, `${tag}: tema ${theme}`);
      if (w !== 768) {
        const axe = await new AxeBuilder({ page }).analyze();
        const bad = axe.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
        check(bad.length === 0, `${tag}: axe ${bad.map((v) => `${v.id} (${v.nodes.length})`).join(", ")}`);
      }
      if (SHOT_DIR) await page.screenshot({ path: `${SHOT_DIR}/${p}-${w}-${scheme === "dark" ? "gelap" : "terang"}.png` });
    }
    await ctx.close();
  }
}

// Interaksi penampil.
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(`${base}#/inspeksi?produk=${summary.classes[0]}`);
  await page.waitForSelector(".stage img");
  const zoom = page.locator(".zoom-tools .mono");
  await page.getByRole("button", { name: "Perbesar" }).click();
  check((await zoom.textContent()) === "150%", "zoom tombol tidak menjadi 150%");
  await page.getByRole("button", { name: "Reset" }).click();
  check((await zoom.textContent()) === "100%", "reset zoom gagal");
  await page.getByLabel("Geser batas antara foto asli dan peta skor").fill("20");
  const clip = await page.locator(".layer.after").getAttribute("style");
  check(/inset\(0px 0px 0px 20%\)|inset\(0 0 0 20%\)/.test(clip ?? ""), `pembanding tidak bergeser: ${clip}`);
  const thumbs = page.locator(".strip .thumb");
  check((await thumbs.count()) >= 8, "strip sampel kurang dari 8");
  await thumbs.nth(1).click();
  check(/foto=/.test(page.url()), "klik sampel tidak mengubah URL");
  await page.close();
}

// Inferensi browser dan paritas.
if (summary.live_method) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  let compared = 0;
  let maxRel = 0;
  const times = [];
  for (const cls of summary.classes.slice(0, Number(process.env.LIVE_CLASSES || summary.classes.length))) {
    const data = JSON.parse(readFileSync(new URL(`../dist/data/samples/${cls}.json`, import.meta.url)));
    const thr = data.thresholds[summary.live_method];
    for (const s of data.items.slice(0, 2)) {
      await page.goto(`${base}#/inspeksi?produk=${cls}&foto=${s.id}`);
      await page.getByRole("button", { name: "Hitung ulang di browser" }).click();
      const el = page.locator("[data-live-score]");
      await el.waitFor({ timeout: 120000 });
      const live = Number(await el.getAttribute("data-live-score"));
      const ref = s.scores[`${summary.live_method}_int8`];
      const rel = Math.abs(live - ref) / thr;
      maxRel = Math.max(maxRel, rel);
      compared++;
      check(rel < 1e-3, `${cls}/${s.id}: skor browser ${live} vs Python ${ref}`);
      check((live > thr) === (ref > thr), `${cls}/${s.id}: keputusan browser berbeda`);
      const t = await page.locator("[data-testid=live-result] dd").nth(3).textContent();
      times.push(t);
    }
  }
  check(errors.length === 0, `inferensi: ${errors.join(" | ")}`);
  console.log(`paritas browser: ${compared} foto, selisih maks ${maxRel.toExponential(2)} x ambang; waktu ${times.join(", ")}`);
  await page.close();
} else {
  console.log("model browser belum diekspor: uji inferensi langsung dilewati");
}

await browser.close();
server.close();
console.log(`${checks} pemeriksaan, ${failures.length} gagal`);
if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
