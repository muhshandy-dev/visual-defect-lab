import { useEffect, useMemo, useState } from "react";
import { href } from "../lib/route.ts";
import { loadSamples, url } from "../lib/data.ts";
import { CLASS_NAMES, METHOD_SHORT, defectName } from "../lib/labels.ts";
import type { ClassSamples, Sample, Summary } from "../lib/types.ts";

type Filter = "semua" | "cacat" | "normal" | "salah";

interface Row {
  cls: string;
  s: Sample;
  thr: number;
}

export function Gallery({ summary }: { summary: Summary }) {
  const [all, setAll] = useState<ClassSamples[] | null>(null);
  const [cls, setCls] = useState("semua");
  const [filter, setFilter] = useState<Filter>("semua");
  const m = summary.primary;

  useEffect(() => {
    Promise.all(summary.classes.map(loadSamples)).then(setAll);
  }, [summary.classes]);

  const rows = useMemo(() => {
    if (!all) return [];
    const out: Row[] = [];
    for (const d of all) {
      if (cls !== "semua" && d.cls !== cls) continue;
      for (const s of d.items) out.push({ cls: d.cls, s, thr: d.thresholds[m] });
    }
    return out.filter(({ s, thr }) => {
      const wrong = s.scores[m] > thr !== (s.label === 1);
      if (filter === "cacat") return s.label === 1;
      if (filter === "normal") return s.label === 0;
      if (filter === "salah") return wrong;
      return true;
    });
  }, [all, cls, filter, m]);

  return (
    <div className="page">
      <div className="page-head">
        <h1>Galeri foto uji</h1>
        <p>
          Setiap kartu membagi foto menjadi dua: kiri foto asli, kanan peta skor {METHOD_SHORT[m]}. Gerakkan kursor di atas kartu untuk menggeser
          batasnya, lalu buka kartu untuk memperbesar dan membandingkan tiga model. Sampel dipilih acak dengan benih tetap, ditambah
          kesalahan terbesar model {METHOD_SHORT[m]} per produk agar kegagalan juga terlihat.
        </p>
      </div>
      <div className="filters">
        <label>
          <span>Produk</span>
          <select value={cls} onChange={(e) => setCls(e.target.value)}>
            <option value="semua">Semua produk</option>
            {summary.classes.map((c) => (
              <option key={c} value={c}>
                {CLASS_NAMES[c]}
              </option>
            ))}
          </select>
        </label>
        <div className="seg-group" role="radiogroup" aria-label="Saring foto">
          {(
            [
              ["semua", "Semua"],
              ["cacat", "Cacat"],
              ["normal", "Normal"],
              ["salah", "Keputusan salah"],
            ] as const
          ).map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={filter === k} className={filter === k ? "seg is-active" : "seg"} onClick={() => setFilter(k)}>
              {label}
            </button>
          ))}
        </div>
        <span className="muted small" aria-live="polite">
          {all ? `${rows.length} foto` : "Memuat"}
        </span>
      </div>
      <ul className="cards">
        {rows.map(({ cls: c, s, thr }) => {
          const rej = s.scores[m] > thr;
          const wrong = rej !== (s.label === 1);
          return (
            <li key={`${c}/${s.id}`}>
              <a
                className="card"
                href={href("inspeksi", c, s.id)}
                onPointerMove={(e) => {
                  const r = e.currentTarget.getBoundingClientRect();
                  e.currentTarget.style.setProperty("--split", `${((e.clientX - r.left) / r.width) * 100}%`);
                }}
                onPointerLeave={(e) => e.currentTarget.style.removeProperty("--split")}
              >
                <span className="card-img">
                  <img src={url(s.input)} alt="" loading="lazy" width={256} height={256} />
                  <img className="card-after" src={url(s.maps[`${m}_thumb`] ?? s.input)} alt="" loading="lazy" width={256} height={256} />
                </span>
                <span className="card-body">
                  <span className="card-title">{CLASS_NAMES[c]}</span>
                  <span className="small">{s.label ? `Cacat: ${defectName(s.defect)}` : "Normal"}</span>
                  <span className="card-tags">
                    <span className={rej ? "tag tag-reject" : "tag"}>{rej ? "Tolak" : "Lolos"}</span>
                    {wrong && <span className="tag tag-wrong">{s.label ? "Terlewat" : "Alarm palsu"}</span>}
                  </span>
                </span>
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
