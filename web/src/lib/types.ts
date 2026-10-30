export type MetricKey = "image_auroc" | "pixel_auroc" | "aupro" | "miou";
export type CI = [number, number];
export type OpKey = "false_reject" | "miss";

export interface ClassMetrics {
  image_auroc: number;
  pixel_auroc: number;
  aupro: number;
  miou: number;
  ci95: Record<MetricKey, CI> & Partial<Record<OpKey, CI>>;
  threshold: number;
  test_false_reject: number;
  test_miss: number;
  test_missed?: number;
  test_defect_images?: number;
  test_false_rejected?: number;
  test_normal_images?: number;
  ms_per_test_image: number;
  bank_size: number;
  fit_images: number;
  val_images: number;
  test_images: number;
}

export interface SizeRow {
  group: string;
  n: number;
  detected: number;
  miou: number;
}

export interface DefectRow {
  defect: string;
  n: number;
  detected: number;
  miou: number;
}

export interface MethodSummary {
  key: string;
  label: string;
  macro: Record<MetricKey, number>;
  macro_ci95: Record<MetricKey, CI> & Partial<Record<OpKey, CI>>;
  macro_false_reject: number;
  macro_miss: number;
  per_class: Record<string, ClassMetrics>;
  by_size: SizeRow[];
  by_defect: Record<string, DefectRow[]>;
}

export interface PairedDiff {
  diff: number;
  ci95: CI;
  p_le_zero: number;
}

export interface OnnxReport {
  backbone: { file: string; bytes: number; fp32_bytes: number; sha256: string };
  head_bytes_total: number;
  head_bytes_max: number;
  macro: Record<string, number>;
  decision_flips_int8_total: number;
  test_images_total: number;
}

export interface Summary {
  generated: string;
  primary: string;
  live_method: string | null;
  dataset: {
    name: string;
    version: string;
    sha256: string;
    licence: string;
    licence_url: string;
    source_url: string;
    citation: string;
    counts: Record<string, number>;
  };
  config: Record<string, unknown>;
  method_order: string[];
  methods: Record<string, MethodSummary>;
  paired: Record<string, Record<MetricKey, PairedDiff>>;
  classes: string[];
  onnx: OnnxReport | null;
}

export interface Sample {
  id: string;
  group: "acak" | "terlewat" | "alarm-palsu";
  label: 0 | 1;
  defect: string;
  area: number;
  img: string;
  w: number;
  h: number;
  input: string;
  mask: string | null;
  maps: Record<string, string>;
  scores: Record<string, number>;
  iou: Record<string, number | null>;
}

export interface ClassSamples {
  cls: string;
  thresholds: Record<string, number>;
  items: Sample[];
}
