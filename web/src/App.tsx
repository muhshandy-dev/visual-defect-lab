import { useEffect, useState } from "react";
import { Errors } from "./components/Errors.tsx";
import { Gallery } from "./components/Gallery.tsx";
import { Inspect } from "./components/Inspect.tsx";
import { Method } from "./components/Method.tsx";
import { Performance } from "./components/Performance.tsx";
import { loadSummary } from "./lib/data.ts";
import { PAGES, href, parse, type Route } from "./lib/route.ts";
import type { Summary } from "./lib/types.ts";

function readTheme(): "gelap" | "terang" {
  try {
    const t = localStorage.getItem("tema");
    if (t === "gelap" || t === "terang") return t;
  } catch {
    /* penyimpanan diblokir: pakai preferensi sistem */
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "terang" : "gelap";
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [theme, setTheme] = useState(readTheme);

  useEffect(() => {
    const on = () => setRoute(parse(location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    loadSummary().then(setSummary, (e) => setError(String(e)));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme === "terang" ? "light" : "dark";
    try {
      localStorage.setItem("tema", theme);
    } catch {
      /* abaikan */
    }
  }, [theme]);

  useEffect(() => {
    const label = PAGES.find(([p]) => p === route.page)?.[1];
    document.title = `${label} · Visual Defect Lab`;
    window.scrollTo(0, 0);
  }, [route.page]);

  const cls = route.cls && summary?.classes.includes(route.cls) ? route.cls : (summary?.classes[0] ?? "");

  return (
    <div className="app">
      <a className="skip" href="#isi">
        Lewati ke isi
      </a>
      <header className="topbar">
        <a className="brand" href={href("inspeksi")}>
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <rect x="2.5" y="2.5" width="19" height="19" rx="2" fill="none" stroke="currentColor" strokeWidth="1.6" />
            <circle cx="15" cy="9" r="3.2" fill="var(--amber)" />
            <path d="M5 18h7M5 14.5h4" stroke="currentColor" strokeWidth="1.6" />
          </svg>
          <span>Visual Defect Lab</span>
        </a>
        <nav aria-label="Menu utama">
          {PAGES.map(([p, label]) => (
            <a key={p} href={href(p, p === "inspeksi" ? route.cls : null)} aria-current={route.page === p ? "page" : undefined}>
              {label}
            </a>
          ))}
        </nav>
        <button
          type="button"
          className="theme"
          onClick={() => setTheme(theme === "gelap" ? "terang" : "gelap")}
          aria-label={theme === "gelap" ? "Pakai tema terang" : "Pakai tema gelap"}
        >
          {theme === "gelap" ? "Terang" : "Gelap"}
        </button>
      </header>

      <main id="isi" tabIndex={-1}>
        {error && <p className="error">Data website gagal dimuat: {error}</p>}
        {!summary && !error && <p className="muted pad">Memuat</p>}
        {summary && route.page === "inspeksi" && (
          <Inspect
            summary={summary}
            cls={cls}
            sampleId={route.sample}
            navigate={(c, s) => {
              location.hash = href("inspeksi", c, s);
            }}
          />
        )}
        {summary && route.page === "galeri" && <Gallery summary={summary} />}
        {summary && route.page === "kinerja" && <Performance summary={summary} />}
        {summary && route.page === "kesalahan" && <Errors summary={summary} />}
        {summary && route.page === "metode" && <Method summary={summary} />}
      </main>

      <footer className="footer">
        <p>
          Foto produk dari dataset VisA (Zou dkk., ECCV 2022), lisensi{" "}
          <a href="https://creativecommons.org/licenses/by/4.0/" rel="license noopener" target="_blank">
            CC BY 4.0
          </a>
          . Foto diperkecil dan dipadatkan untuk website ini. Kode:{" "}
          <a href="https://github.com/muhshandy-dev/visual-defect-lab" rel="noopener" target="_blank">
            github.com/muhshandy-dev/visual-defect-lab
          </a>
          .
        </p>
        <p className="muted">Demo portofolio. Bukan sistem inspeksi produksi.</p>
      </footer>
    </div>
  );
}
