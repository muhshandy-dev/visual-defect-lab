import { useState } from "react";
import { bytes, ci, num, pct, signed, pctCi } from "../lib/format.ts";
import { CLASS_NAMES, METHOD_SHORT } from "../lib/labels.ts";
import type { MetricKey, Summary } from "../lib/types.ts";

export const METRICS: Array<[MetricKey, string, string]> = [
  ["image_auroc", "Pemisahan foto (image AUROC)", "Seberapa baik skor membedakan foto cacat dari foto normal. 1 berarti sempurna, 0,5 sama dengan menebak."],
  ["pixel_auroc", "Pemisahan piksel (pixel AUROC)", "Hal yang sama, tetapi per piksel. Angkanya cenderung tinggi karena sebagian besar piksel normal mudah dikenali."],
  ["aupro", "Cakupan area cacat (AUPRO)", "Rata-rata bagian tiap area cacat yang tertandai, selama piksel normal yang ikut tertandai paling banyak 30%. Lebih adil untuk cacat kecil."],
  ["miou", "Kecocokan area (mIoU)", "Tumpang tindih area yang ditandai model dengan area cacat sebenarnya, pada ambang operasi. 1 berarti persis sama."],
];

function share(v: number, lo: number, hi: number): string {
  const t = hi > lo ? Math.min(Math.max((v - lo) / (hi - lo), 0), 1) : 1;
  return `${Math.round(6 + t * 94)}%`;
}

export function Performance({ summary }: { summary: Summary }) {
  const [metric, setMetric] = useState<MetricKey>("image_auroc");
  const ms = summary.method_order;
  const vals = summary.classes.flatMap((c) => ms.map((m) => summary.methods[m].per_class[c][metric]));
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const onnx = summary.onnx;

  return (
    <div className="page">
      <div className="page-head">
        <h1>Kinerja model</h1>
        <p>
          Semua angka dihitung pada {summary.dataset.counts.test_normal + summary.dataset.counts.test_defect} foto uji resmi VisA (
          {summary.dataset.counts.test_defect} cacat, {summary.dataset.counts.test_normal} normal) dari 12 produk, lalu dirata-rata per produk.
          Rentang di bawah tiap angka adalah selang kepercayaan 95% dari {String(summary.config.bootstrap_reps)} kali pengambilan sampel ulang foto uji.
        </p>
      </div>

      <section className="block">
        <h2>Ringkasan tiga model</h2>
        <div className="table-wrap" tabIndex={0}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Ukuran</th>
                {ms.map((m) => (
                  <th key={m} scope="col">
                    {METHOD_SHORT[m]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS.map(([k, label, desc]) => (
                <tr key={k}>
                  <th scope="row">
                    {label}
                    <span className="muted small sub">{desc}</span>
                  </th>
                  {ms.map((m) => (
                    <td key={m}>
                      <span className="big mono">{num(summary.methods[m].macro[k])}</span>
                      <span className="muted small sub">{ci(summary.methods[m].macro_ci95[k])}</span>
                    </td>
                  ))}
                </tr>
              ))}
              <tr>
                <th scope="row">
                  Foto normal yang ditolak
                  <span className="muted small sub">Alarm palsu pada ambang operasi. Ambang dipilih agar sekitar 5% foto normal validasi ditolak.</span>
                </th>
                {ms.map((m) => (
                  <td key={m}>
                    <span className="mono">{pct(summary.methods[m].macro_false_reject, 1)}</span>
                    <span className="muted small sub">{pctCi(summary.methods[m].macro_ci95.false_reject, 1)}</span>
                  </td>
                ))}
              </tr>
              <tr>
                <th scope="row">
                  Foto cacat yang lolos
                  <span className="muted small sub">Cacat terlewat pada ambang yang sama.</span>
                </th>
                {ms.map((m) => (
                  <td key={m}>
                    <span className="mono">{pct(summary.methods[m].macro_miss, 1)}</span>
                    <span className="muted small sub">{pctCi(summary.methods[m].macro_ci95.miss, 1)}</span>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {Object.keys(summary.paired).length > 0 && (
        <section className="block">
          <h2>Selisih berpasangan</h2>
          <p>
            Kedua model dinilai pada sampel ulang yang sama, jadi selisihnya bisa diuji langsung. Bila seluruh rentang di atas nol, keunggulan itu bukan
            kebetulan sampel.
          </p>
          <div className="table-wrap" tabIndex={0}>
            <table className="data">
              <thead>
                <tr>
                  <th scope="col">Perbandingan</th>
                  {METRICS.map(([k, label]) => (
                    <th key={k} scope="col">
                      {label.split(" (")[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.paired).map(([key, d]) => {
                  const [a, , b] = key.split("_");
                  return (
                    <tr key={key}>
                      <th scope="row">
                        {METHOD_SHORT[a]} dibanding {METHOD_SHORT[b]}
                      </th>
                      {METRICS.map(([k]) => (
                        <td key={k}>
                          <span className="mono">{signed(d[k].diff)}</span>
                          <span className="muted small sub">{ci(d[k].ci95)}</span>
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <section className="block">
        <h2>Per produk</h2>
        <div className="seg-group" role="radiogroup" aria-label="Pilih ukuran">
          {METRICS.map(([k, label]) => (
            <button key={k} type="button" role="radio" aria-checked={metric === k} className={metric === k ? "seg is-active" : "seg"} onClick={() => setMetric(k)}>
              {label.split(" (")[0]}
            </button>
          ))}
        </div>
        <div className="table-wrap" tabIndex={0}>
          <table className="data heat-table">
            <caption className="muted small">Panjang batang dibandingkan dengan nilai terendah dan tertinggi di tabel ini.</caption>
            <thead>
              <tr>
                <th scope="col">Produk</th>
                {ms.map((m) => (
                  <th key={m} scope="col">
                    {METHOD_SHORT[m]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {summary.classes.map((c) => (
                <tr key={c}>
                  <th scope="row">{CLASS_NAMES[c]}</th>
                  {ms.map((m) => {
                    const r = summary.methods[m].per_class[c];
                    return (
                      <td key={m}>
                        <span className="mono">{num(r[metric])}</span>
                        <span className="cellbar" aria-hidden="true">
                          <span style={{ width: share(r[metric], lo, hi) }} />
                        </span>
                        <span className="small sub">{ci(r.ci95[metric], 2)}</span>
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
        <h2>Ukuran dan kecepatan</h2>
        <div className="table-wrap" tabIndex={0}>
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Waktu per foto (CPU 4 inti, Python)</th>
                <th scope="col">Patch normal dalam memori (rata-rata per produk)</th>
              </tr>
            </thead>
            <tbody>
              {ms.map((m) => {
                const pc = Object.values(summary.methods[m].per_class);
                const avg = (f: (x: (typeof pc)[number]) => number) => pc.reduce((a, x) => a + f(x), 0) / pc.length;
                return (
                  <tr key={m}>
                    <th scope="row">{METHOD_SHORT[m]}</th>
                    <td className="mono">{num(avg((x) => x.ms_per_test_image), 0)} ms</td>
                    <td className="mono">{num(avg((x) => x.bank_size), 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {onnx && (
          <>
            <h3>Model browser setelah kuantisasi int8</h3>
            <p>
              Backbone ResNet18 dipadatkan ke int8 ({bytes(onnx.backbone.bytes)}, dari {bytes(onnx.backbone.fp32_bytes)}).{" "}
              {onnx.backbone.variant === "weights"
                ? "Hanya bobot yang dibulatkan ke int8; perhitungan di browser tetap float32."
                : "Bobot dan aktivasi dibulatkan ke int8."}{" "}
              {onnx.selection &&
                `Cara kuantisasi dipilih dari kesetiaan skor pada foto normal validasi, bukan dari foto uji: selisih maksimum ${num(onnx.selection.val_max_rel_diff[onnx.backbone.variant ?? "weights"], 3)} kali ambang (batas ${num(onnx.selection.fidelity_limit, 2)}).`}{" "}
              Memori patch normal tiap produk
              disimpan sebagai berkas terpisah sekitar {bytes(onnx.head_bytes_max)} yang diunduh hanya saat produk itu dihitung. Pada seluruh{" "}
              {onnx.test_images_total} foto uji, kuantisasi mengubah pemisahan foto dari {num(onnx.macro.image_auroc_torch)} menjadi{" "}
              {num(onnx.macro.image_auroc_int8)} dan pemisahan piksel dari {num(onnx.macro.pixel_auroc_torch)} menjadi {num(onnx.macro.pixel_auroc_int8)}.
              Keputusan lolos atau tolak berubah pada {onnx.decision_flips_int8_total} foto.
            </p>
          </>
        )}
      </section>
    </div>
  );
}
