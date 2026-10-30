const nf = (d: number) =>
  new Intl.NumberFormat("id-ID", { minimumFractionDigits: d, maximumFractionDigits: d });

export function num(v: number | null | undefined, d = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "tidak ada";
  return nf(d).format(v);
}

export function pct(v: number | null | undefined, d = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "tidak ada";
  return `${nf(d).format(v * 100)}%`;
}

export function ci(c: [number, number] | undefined, d = 3): string {
  if (!c) return "";
  return `${num(c[0], d)} sampai ${num(c[1], d)}`;
}

export function pctCi(c: [number, number] | undefined, d = 0): string {
  if (!c) return "";
  return `${nf(d).format(c[0] * 100)} sampai ${pct(c[1], d)}`;
}

export function signed(v: number, d = 3): string {
  const s = num(Math.abs(v), d);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

export function bytes(n: number): string {
  if (n >= 1024 * 1024) return `${nf(1).format(n / (1024 * 1024))} MB`;
  return `${nf(0).format(n / 1024)} KB`;
}
