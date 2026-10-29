// Pemeriksaan hasil build: ukuran berkas di bawah batas Cloudflare Pages dan tanpa URL pihak ketiga.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const dist = new URL("../dist/", import.meta.url).pathname;
const LIMIT = 25 * 1024 * 1024;
let files = 0;
let largest = ["", 0];
const problems = [];
function walk(dir) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p);
    else {
      files++;
      if (st.size > largest[1]) largest = [p, st.size];
      if (st.size > LIMIT) problems.push(`${p} melebihi 25 MiB`);
      if (/\.(html|js|css)$/.test(name)) {
        const txt = readFileSync(p, "utf8");
        const ext = txt.match(/(?:src|href)=["']https?:\/\/[^"']+/g) ?? [];
        for (const e of ext) if (!/github\.com|creativecommons\.org/.test(e)) problems.push(`${p}: ${e}`);
      }
    }
  }
}
walk(dist);
if (files > 20000) problems.push(`terlalu banyak berkas: ${files}`);
console.log(`${files} berkas; terbesar ${largest[0]} (${(largest[1] / 1048576).toFixed(1)} MB)`);
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
