import { num } from "../lib/format.ts";

/** Batang skor relatif terhadap ambang: 0 sampai 2 kali ambang, penanda di 1. */
export function ScoreBar({ ratio }: { ratio: number }) {
  const pos = Math.min(Math.max(ratio / 2, 0), 1) * 100;
  return (
    <div
      className="scorebar"
      role="meter"
      aria-valuemin={0}
      aria-valuemax={2}
      aria-valuenow={Number(ratio.toFixed(2))}
      aria-label={`Skor ${num(ratio, 2)} kali ambang`}
    >
      <div className="scorebar-fill" style={{ width: `${pos}%` }} data-over={ratio > 1 ? "true" : "false"} />
      <div className="scorebar-mark" style={{ left: "50%" }} />
      <div className="scorebar-scale" aria-hidden="true">
        <span>0</span>
        <span>ambang</span>
        <span>2x</span>
      </div>
    </div>
  );
}
