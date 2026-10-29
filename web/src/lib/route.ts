export const PAGES = [
  ["inspeksi", "Inspeksi"],
  ["galeri", "Galeri"],
  ["kinerja", "Kinerja"],
  ["kesalahan", "Kesalahan"],
  ["metode", "Metode"],
] as const;
export type Page = (typeof PAGES)[number][0];

export interface Route {
  page: Page;
  cls: string | null;
  sample: string | null;
}

export function parse(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#\/?/, "").split("?");
  const q = new URLSearchParams(query);
  const page = (PAGES.find(([p]) => p === path)?.[0] ?? "inspeksi") as Page;
  return { page, cls: q.get("produk"), sample: q.get("foto") };
}

export function href(page: Page, cls?: string | null, sample?: string | null): string {
  const q = new URLSearchParams();
  if (cls) q.set("produk", cls);
  if (sample) q.set("foto", sample);
  const s = q.toString();
  return `#/${page}${s ? `?${s}` : ""}`;
}
