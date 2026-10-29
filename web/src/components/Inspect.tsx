import { useEffect, useMemo, useState } from "react";
import { loadGray, loadSamples, url } from "../lib/data.ts";
import { num as format } from "../lib/format.ts";
import { quantizeMap, verdict } from "../lib/heat.ts";
import { imageToRGBA, runLive, type LiveResult } from "../lib/infer.ts";
import { CLASS_NAMES, CLASS_NOTES, METHOD_DESC, METHOD_SHORT, defectName } from "../lib/labels.ts";
import type { ClassSamples, Sample, Summary } from "../lib/types.ts";
import { ScoreBar } from "./ScoreBar.tsx";
import { type Layers, Viewer } from "./Viewer.tsx";

interface Props {
  summary: Summary;
  cls: string;
  sampleId: string | null;
  navigate: (cls: string, sampleId: string | null) => void;
}

interface Live {
  key: string;
  result?: LiveResult;
  error?: string;
  running: boolean;
}

interface Upload {
  src: string;
  w: number;
  h: number;
  name: string;
  result?: LiveResult;
  error?: string;
  running: boolean;
}

const GROUP_LABEL: Record<Sample["group"], string> = {
  acak: "Sampel acak",
  terlewat: "Cacat terlewat",
  "alarm-palsu": "Alarm palsu",
};

export function Inspect({ summary, cls, sampleId, navigate }: Props) {
  const [data, setData] = useState<ClassSamples | null>(null);
  const [method, setMethod] = useState(summary.primary);
  const [layers, setLayers] = useState<Layers>({ heat: true, detected: true, truth: true });
  const [map, setMap] = useState<Uint8Array | null>(null);
  const [mask, setMask] = useState<Uint8Array | null>(null);
  const [live, setLive] = useState<Live | null>(null);
  const [upload, setUpload] = useState<Upload | null>(null);
  const liveMethod = summary.live_method;

  useEffect(() => {
    let off = false;
    setData(null);
    loadSamples(cls).then((d) => !off && setData(d));
    return () => {
      off = true;
    };
  }, [cls]);

  const sample = useMemo(() => {
    if (!data) return null;
    return data.items.find((s) => s.id === sampleId) ?? data.items[0];
  }, [data, sampleId]);

  const liveKey = sample ? `${cls}/${sample.id}` : "";
  const liveHere = live && live.key === liveKey && live.result ? live.result : null;
  const threshold = data?.thresholds[method] ?? 1;

  useEffect(() => {
    let off = false;
    if (upload) {
      setMask(null);
      setMap(upload.result && liveMethod ? quantizeMap(upload.result.map, data!.thresholds[liveMethod]) : null);
      return;
    }
    if (!sample) return;
    if (method === liveMethod && liveHere) {
      setMap(quantizeMap(liveHere.map, threshold));
    } else {
      setMap(null);
      loadGray(sample.maps[method]).then((m) => !off && setMap(m));
    }
    if (sample.mask) loadGray(sample.mask).then((m) => !off && setMask(m));
    else setMask(null);
    return () => {
      off = true;
    };
  }, [sample, method, liveHere, threshold, upload, liveMethod, data]);

  async function startLive() {
    if (!sample || !liveMethod) return;
    setMethod(liveMethod);
    setLive({ key: liveKey, running: true });
    try {
      const rgba = await imageToRGBA(url(sample.input));
      const result = await runLive(rgba, cls);
      setLive({ key: liveKey, running: false, result });
    } catch (e) {
      setLive({ key: liveKey, running: false, error: String(e) });
    }
  }

  async function onFile(file: File) {
    if (!liveMethod) return;
    const src = URL.createObjectURL(file);
    const bmp = await createImageBitmap(file);
    const base = { src, w: bmp.width, h: bmp.height, name: file.name };
    bmp.close();
    setUpload({ ...base, running: true });
    try {
      const result = await runLive(await imageToRGBA(file), cls);
      setUpload({ ...base, running: false, result });
    } catch (e) {
      setUpload({ ...base, running: false, error: String(e) });
    }
  }

  function closeUpload() {
    if (upload) URL.revokeObjectURL(upload.src);
    setUpload(null);
  }

  const shownScore = sample
    ? method === liveMethod && liveHere
      ? liveHere.score
      : sample.scores[method]
    : 0;
  const v = verdict(shownScore, threshold);
  const correct = sample ? (v === "tolak") === (sample.label === 1) : false;

  return (
    <div className="inspect">
      <aside className="products" aria-label="Pilih produk">
        <h2 className="panel-title">Produk</h2>
        <ul>
          {summary.classes.map((c) => (
            <li key={c}>
              <button
                type="button"
                className={c === cls ? "product is-active" : "product"}
                aria-current={c === cls ? "true" : undefined}
                onClick={() => {
                  closeUpload();
                  navigate(c, null);
                }}
              >
                <span className="product-name">{CLASS_NAMES[c]}</span>
                <span className="product-note">{CLASS_NOTES[c]}</span>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <section className="bench" aria-label="Meja inspeksi">
        <div className="bench-head">
          <div>
            <h1>{CLASS_NAMES[cls]}</h1>
            <p className="muted">
              {upload
                ? `Foto unggahan: ${upload.name}. Dibandingkan dengan produk normal ${CLASS_NAMES[cls]}.`
                : sample
                  ? `${GROUP_LABEL[sample.group]}. Foto uji yang tidak pernah dilihat model saat belajar.`
                  : "Memuat sampel"}
            </p>
          </div>
          <div className="method-switch" role="radiogroup" aria-label="Model yang ditampilkan">
            {summary.method_order.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={method === m}
                className={method === m ? "seg is-active" : "seg"}
                onClick={() => setMethod(m)}
                disabled={!!upload && m !== liveMethod}
              >
                {METHOD_SHORT[m]}
              </button>
            ))}
          </div>
        </div>

        {upload ? (
          <Viewer img={upload.src} w={upload.w} h={upload.h} alt={`Foto unggahan ${upload.name}`} map={map} mask={null} layers={layers} />
        ) : sample ? (
          <Viewer
            img={url(sample.img)}
            w={sample.w}
            h={sample.h}
            alt={`${CLASS_NAMES[cls]}, ${sample.label ? `cacat: ${defectName(sample.defect)}` : "normal"}`}
            map={map}
            mask={mask}
            layers={layers}
          />
        ) : (
          <div className="stage skeleton" style={{ aspectRatio: "4 / 3" }} />
        )}

        <fieldset className="layer-toggles">
          <legend>Lapisan</legend>
          <label>
            <input type="checkbox" checked={layers.heat} onChange={(e) => setLayers({ ...layers, heat: e.target.checked })} />
            <span className="swatch swatch-heat" aria-hidden="true" /> Peta skor
          </label>
          <label>
            <input type="checkbox" checked={layers.detected} onChange={(e) => setLayers({ ...layers, detected: e.target.checked })} />
            <span className="swatch swatch-detect" aria-hidden="true" /> Area yang ditandai model
          </label>
          <label>
            <input type="checkbox" checked={layers.truth} onChange={(e) => setLayers({ ...layers, truth: e.target.checked })} disabled={!!upload} />
            <span className="swatch swatch-truth" aria-hidden="true" /> Cacat sebenarnya
          </label>
        </fieldset>

        {data && !upload && (
          <div className="strip" role="list" aria-label="Foto uji">
            {data.items.map((s) => (
              <button
                key={s.id}
                type="button"
                role="listitem"
                className={s.id === sample?.id ? "thumb is-active" : "thumb"}
                onClick={() => navigate(cls, s.id)}
                aria-label={`${s.label ? "Cacat" : "Normal"}, ${GROUP_LABEL[s.group]}`}
                aria-current={s.id === sample?.id ? "true" : undefined}
              >
                <img src={url(s.input)} alt="" loading="lazy" width={64} height={64} />
                <span className={`thumb-tag ${s.label ? "is-defect" : ""}`}>{s.label ? "Cacat" : "Normal"}</span>
                {s.group !== "acak" && <span className="thumb-flag">{s.group === "terlewat" ? "Terlewat" : "Alarm palsu"}</span>}
              </button>
            ))}
          </div>
        )}
      </section>

      <aside className="verdict-panel" aria-label="Hasil pemeriksaan">
        {upload ? (
          <UploadPanel upload={upload} threshold={liveMethod ? data?.thresholds[liveMethod] ?? 1 : 1} onClose={closeUpload} />
        ) : sample && data ? (
          <>
            <div className={`verdict verdict-${v}`} aria-live="polite">
              <span className="verdict-label">Keputusan {METHOD_SHORT[method]}</span>
              <strong>{v === "tolak" ? "Tolak: cacat terdeteksi" : "Lolos: tidak ada cacat"}</strong>
              <ScoreBar ratio={shownScore / threshold} />
              <span className="muted small">
                Skor {format(shownScore, 2)} dibanding ambang {format(threshold, 2)}. Skor di atas ambang berarti ditolak.
              </span>
            </div>

            <dl className="facts">
              <div>
                <dt>Label sebenarnya</dt>
                <dd>{sample.label ? `Cacat: ${defectName(sample.defect)}` : "Normal"}</dd>
              </div>
              <div>
                <dt>Keputusan ini</dt>
                <dd className={correct ? "" : "is-wrong"}>{correct ? "Benar" : sample.label ? "Salah: cacat terlewat" : "Salah: alarm palsu"}</dd>
              </div>
              {sample.label === 1 && (
                <div>
                  <dt>Luas cacat</dt>
                  <dd>{format(sample.area * 100, 2)}% dari foto</dd>
                </div>
              )}
              {sample.label === 1 && method !== liveMethod && sample.iou[method] !== null && (
                <div>
                  <dt>Kecocokan area (IoU)</dt>
                  <dd>{format(sample.iou[method] ?? 0, 2)}</dd>
                </div>
              )}
            </dl>

            <h3 className="panel-title">Tiga model pada foto ini</h3>
            <table className="mini">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Skor / ambang</th>
                  <th scope="col">Keputusan</th>
                </tr>
              </thead>
              <tbody>
                {summary.method_order.map((m) => {
                  const r = sample.scores[m] / data.thresholds[m];
                  return (
                    <tr key={m} className={m === method ? "is-active" : ""}>
                      <th scope="row">{METHOD_SHORT[m]}</th>
                      <td className="mono">{format(r, 2)}</td>
                      <td>{r > 1 ? "Tolak" : "Lolos"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="muted small">{METHOD_DESC[method]}</p>

            {liveMethod && (
              <div className="live">
                <h3 className="panel-title">Jalankan di perangkat ini</h3>
                <p className="small">
                  Model {METHOD_SHORT[liveMethod]} (int8, ONNX) berjalan di browser Anda. Foto tidak dikirim ke server mana pun.
                </p>
                <button type="button" className="primary" onClick={startLive} disabled={live?.running}>
                  {live?.running && live.key === liveKey ? "Menghitung" : "Hitung ulang di browser"}
                </button>
                {live && live.key === liveKey && live.error && <p className="error">Gagal: {live.error}</p>}
                {liveHere && (
                  <dl className="facts compact" data-testid="live-result">
                    <div>
                      <dt>Skor langsung</dt>
                      <dd className="mono" data-live-score={liveHere.score}>{format(liveHere.score, 4)}</dd>
                    </div>
                    {sample.scores[`${liveMethod}_int8`] !== undefined && (
                      <div>
                        <dt>Model int8 yang sama di Python</dt>
                        <dd className="mono">{format(sample.scores[`${liveMethod}_int8`], 4)}</dd>
                      </div>
                    )}
                    {sample.scores[`${liveMethod}_int8`] !== undefined && (
                      <div>
                        <dt>Selisih browser dan Python</dt>
                        <dd className="mono">{format(Math.abs(liveHere.score - sample.scores[`${liveMethod}_int8`]), 4)}</dd>
                      </div>
                    )}
                    <div>
                      <dt>Model evaluasi (float32, Python)</dt>
                      <dd className="mono">{format(sample.scores[liveMethod], 4)}</dd>
                    </div>
                    <div>
                      <dt>Waktu</dt>
                      <dd className="mono" data-testid="live-time">
                        {format(liveHere.msBackbone + liveHere.msHead, 0)} ms
                        {liveHere.msLoad > 50 ? ` (+${format(liveHere.msLoad, 0)} ms memuat model)` : ""}
                      </dd>
                    </div>
                  </dl>
                )}
                <label className="file">
                  <span className="file-title">Coba foto sendiri</span>
                  <span className="file-button">Pilih foto</span>
                  <input
                    className="visually-hidden"
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) onFile(f);
                      e.target.value = "";
                    }}
                  />
                </label>
                <p className="muted small">
                  Foto akan dibandingkan dengan produk normal {CLASS_NAMES[cls]}. Model hanya mengenal 12 produk VisA, jadi foto benda lain akan tampak "cacat" di mana-mana.
                </p>
              </div>
            )}
          </>
        ) : (
          <p className="muted">Memuat</p>
        )}
      </aside>
    </div>
  );
}

function UploadPanel({ upload, threshold, onClose }: { upload: Upload; threshold: number; onClose: () => void }) {
  const r = upload.result;
  const v = r ? verdict(r.score, threshold) : null;
  return (
    <>
      <div className={`verdict ${v ? `verdict-${v}` : ""}`} aria-live="polite">
        <span className="verdict-label">Foto unggahan, dihitung di browser</span>
        <strong>{upload.running ? "Menghitung" : v === "tolak" ? "Tolak: tampak berbeda dari normal" : v ? "Lolos: mirip produk normal" : "Gagal"}</strong>
        {r && <ScoreBar ratio={r.score / threshold} />}
        {r && (
          <span className="muted small">
            Skor {format(r.score, 2)} dibanding ambang {format(threshold, 2)}. Waktu {format(r.msBackbone + r.msHead, 0)} ms.
          </span>
        )}
      </div>
      {upload.error && <p className="error">Gagal: {upload.error}</p>}
      <p className="small">
        Tidak ada label sebenarnya untuk foto ini, jadi hasilnya tidak bisa dinilai benar atau salah. Foto diperkecil ke 256 x 256 piksel oleh browser; hasilnya bisa sedikit berbeda dari pemrosesan Python.
      </p>
      <button type="button" onClick={onClose}>
        Kembali ke foto uji
      </button>
    </>
  );
}
