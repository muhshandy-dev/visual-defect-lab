// Nama produk dan terjemahan label cacat VisA (label asli berbahasa Inggris, dipisah koma).

export const CLASS_NAMES: Record<string, string> = {
  candle: "Lilin",
  capsules: "Kapsul",
  cashew: "Kacang mete",
  chewinggum: "Permen karet",
  fryum: "Kerupuk fryum",
  macaroni1: "Makaroni 1",
  macaroni2: "Makaroni 2",
  pcb1: "PCB 1",
  pcb2: "PCB 2",
  pcb3: "PCB 3",
  pcb4: "PCB 4",
  pipe_fryum: "Kerupuk pipa",
};

export const CLASS_NOTES: Record<string, string> = {
  candle: "Empat lilin dalam satu foto",
  capsules: "Banyak kapsul hijau, posisi acak",
  cashew: "Satu kacang, posisi bervariasi",
  chewinggum: "Satu permen karet",
  fryum: "Satu kerupuk roda",
  macaroni1: "Empat makaroni, latar hijau polos",
  macaroni2: "Empat makaroni, latar hijau bertekstur",
  pcb1: "Modul sensor ultrasonik",
  pcb2: "Papan sirkuit dengan IC dan pin",
  pcb3: "Modul sensor inframerah",
  pcb4: "Modul pengisi daya baterai",
  pipe_fryum: "Satu kerupuk pipa",
};

const DEFECTS: Record<string, string> = {
  bent: "kaki komponen bengkok",
  "breakage down the middle": "patah di tengah",
  bubble: "gelembung",
  burnt: "gosong",
  "chip around edge and corner": "gompal di tepi dan sudut",
  "chunk of gum missing": "sebagian permen hilang",
  "chunk of wax missing": "sebagian lilin hilang",
  "color spot similar to the object": "noda warna mirip produk",
  "corner and edge breakage": "patah di sudut dan tepi",
  "corner missing": "sudut hilang",
  "corner or edge breakage": "patah di sudut atau tepi",
  damage: "rusak",
  "damaged corner of packaging": "sudut kemasan rusak",
  "different color spot": "noda warna berbeda",
  "different colour spot": "noda warna berbeda",
  dirt: "kotoran",
  discolor: "perubahan warna",
  extra: "komponen berlebih",
  "extra wax in candle": "lilin berlebih",
  "foreign particals on candle": "partikel asing pada lilin",
  "fryum stuck together": "kerupuk saling menempel",
  leak: "bocor",
  melt: "meleleh",
  "middle breakage": "patah di tengah",
  misshape: "bentuk tidak sesuai",
  missing: "komponen hilang",
  other: "lainnya",
  "same colour spot": "noda warna sama",
  scratch: "goresan",
  scratches: "goresan",
  "similar colour spot": "noda warna mirip produk",
  "small chip around edge": "gompal kecil di tepi",
  "small cracks": "retak kecil",
  "small holes": "lubang kecil",
  "small scratches": "goresan kecil",
  "stuck together": "saling menempel",
  "wax melded out of the candle": "lilin meleber keluar",
  "weird candle wick": "sumbu lilin janggal",
  "wrong place": "salah posisi",
};

export function defectName(raw: string): string {
  if (!raw) return "tidak ada cacat";
  const parts = raw.split(",").map((t) => t.trim());
  const names = parts.map((t) => DEFECTS[t] ?? t);
  return [...new Set(names)].join(", ");
}

export const METHOD_SHORT: Record<string, string> = {
  wrn50: "WideResNet50",
  resnet18: "ResNet18",
  classical: "Klasik",
};

export const METHOD_DESC: Record<string, string> = {
  wrn50: "Transfer learning, model paling akurat. Hasil dihitung lebih dulu di Python.",
  resnet18: "Transfer learning, model kecil yang bisa berjalan langsung di browser.",
  classical: "Tanpa deep learning: tekstur, gradien, dan warna dari OpenCV.",
};
