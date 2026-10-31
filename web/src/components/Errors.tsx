import { useEffect, useState } from "react";
import { loadSamples, url } from "../lib/data.ts";
import { num, pct, pctCi } from "../lib/format.ts";
import { CLASS_NAMES, METHOD_SHORT, defectName } from "../lib/labels.ts";
import { href } from "../lib/route.ts";
import type { ClassSamples, Summary } from "../lib/types.ts";

export function Errors({ summary }: { summary: Summary }) {
  const m = summary.primary;
  const ms = summary.method_order;
  const [cls, setCls] = useState(summary.classes[0]);
  const [all, setAll] = useState<ClassSamples[] | null>(null);

  useEffect(() => {
    Promise.all(summary.classes.map(loadSamples)).then(setAll);
  }, [summary.classes]);

  const ranked = [...summary.classes].sort(
    (a, b) => summary.methods[m].per_class[a].image_auroc - summary.methods[m].per_class[b].image_auroc,
  );
  const failures = (all ?? []).flatMap((d) => d.items.filter((s) => s.group !== "acak").map((s) => ({ cls: d.cls, s, thr: d.thresholds[m] })));

  return (
    <div className="page">
      <div className="page-head">
        <h1>Di mana model salah</h1>
        <p>
          Bagian ini memakai model {METHOD_SHORT[m]} kecuali disebut lain. Kesalahan dihitung pada ambang operasi yang dipilih dari foto normal validasi,
          bukan dari foto uji.
        </p>
      </div>

      <section className="block">
        <h2>Produk dari yang tersulit</h2>
        <div className="table-wrap" tabIndex={0}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Produk</th>
                <th scope="col">Pemisahan foto</th>
                <th scope="col">Cacat lolos</th>
                <th scope="col">Normal ditolak</th>
                <th scope="col">Kecocokan area</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((c) => {
                const r = summary.methods[m].per_class[c];
                return (
                  <tr key={c}>
                    <th scope="row">{CLASS_NAMES[c]}</th>
                    <td className="mono">{num(r.image_auroc)}</td>
                    <td>
                      <span className="mono">{pct(r.test_miss)}</span>
                      {r.test_defect_images !== undefined && (
                        <span className="muted small sub">
                          {r.test_missed}/{r.test_defect_images}, CI {pctCi(r.ci95.miss)}
                        </span>
                      )}
                    </td>
                    <td>
                      <span className="mono">{pct(r.test_false_reject)}</span>
                      {r.test_normal_images !== undefined && (
                        <span className="muted small sub">
                          {r.test_false_rejected}/{r.test_normal_images}, CI {pctCi(r.ci95.false_reject)}
                        </span>
                      )}
                    </td>
                    <td className="mono">{num(r.miou, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="muted small">
          Angka kecil di bawah persentase: jumlah foto (salah per total) dan rentang kepercayaan 95% dari bootstrap 1000 kali.
        </p>
      </section>

      <section className="block">
        <h2>Menurut ukuran cacat</h2>
        <p>Ukuran adalah luas area cacat dibanding luas foto. Batas kelompok ditetapkan sebelum melihat skor model.</p>
        <div className="table-wrap" tabIndex={0}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Ukuran cacat</th>
                <th scope="col">Jumlah foto</th>
                {ms.map((k) => (
                  <th key={k} scope="col">
                    {METHOD_SHORT[k]}: tertangkap / kecocokan area
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.methods[m].by_size.map((row, i) => (
                <tr key={row.group}>
                  <th scope="row">{row.group}</th>
                  <td className="mono">{row.n}</td>
                  {ms.map((k) => {
                    const r = summary.methods[k].by_size[i];
                    return (
                      <td key={k} className="mono">
                        {pct(r.detected)} / {num(r.miou, 2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block">
        <h2>Menurut jenis cacat</h2>
        <label className="inline">
          <span>Produk</span>
          <select value={cls} onChange={(e) => setCls(e.target.value)}>
            {summary.classes.map((c) => (
              <option key={c} value={c}>
                {CLASS_NAMES[c]}
              </option>
            ))}
          </select>
        </label>
        <div className="table-wrap" tabIndex={0}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Jenis cacat</th>
                <th scope="col">Jumlah</th>
                {ms.map((k) => (
                  <th key={k} scope="col">
                    {METHOD_SHORT[k]}: tertangkap
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.methods[m].by_defect[cls].map((row, i) => (
                <tr key={row.defect}>
                  <th scope="row">{defectName(row.defect)}</th>
                  <td className="mono">{row.n}</td>
                  {ms.map((k) => (
                    <td key={k} className="mono">
                      {pct(summary.methods[k].by_defect[cls][i].detected)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="block">
        <h2>Contoh kesalahan terbesar</h2>
        <p>
          Per produk diambil paling banyak dua cacat yang paling jauh di bawah ambang (terlewat) dan dua foto normal yang paling jauh di atas ambang (alarm
          palsu). Produk tanpa kesalahan seperti itu tidak muncul.
        </p>
        <ul className="cards">
          {failures.map(({ cls: c, s, thr }) => (
            <li key={`${c}/${s.id}`}>
              <a className="card" href={href("inspeksi", c, s.id)}>
                <span className="card-img">
                  <img src={url(s.input)} alt="" loading="lazy" width={256} height={256} />
                  <img className="card-after" src={url(s.maps[`${m}_thumb`] ?? s.input)} alt="" loading="lazy" width={256} height={256} />
                </span>
                <span className="card-body">
                  <span className="card-title">{CLASS_NAMES[c]}</span>
                  <span className="small">{s.label ? `Cacat: ${defectName(s.defect)}` : "Normal"}</span>
                  <span className="card-tags">
                    <span className="tag tag-wrong">{s.group === "terlewat" ? "Terlewat" : "Alarm palsu"}</span>
                    <span className="tag mono">{num(s.scores[m] / thr, 2)} x ambang</span>
                  </span>
                </span>
              </a>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
