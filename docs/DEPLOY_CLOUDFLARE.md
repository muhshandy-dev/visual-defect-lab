# Deploy ke Cloudflare Pages

Website dibangun dengan Vite (React + TypeScript). Cloudflare Pages menjalankan build dari folder `web`.

## Langkah (di dashboard Cloudflare milik pemilik repo)

1. Workers and Pages, Create, Pages, Connect to Git, pilih `visual-defect-lab`.
2. Project name: `visual-defect-lab`.
3. Production branch: `main`.
4. Root directory (advanced): `web`.
5. Build command: `npm ci && npm run build`. Build output directory: `dist`.
6. Environment variable: `NODE_VERSION` = `22`.
7. Simpan dan deploy. URL: `https://visual-defect-lab.pages.dev`.
8. Opsional: matikan preview build untuk branch non-produksi (Settings, Builds). Kuota 500 build per bulan dipakai bersama semua proyek.

## Catatan

- `web/public/_headers` memasang CSP ketat. Satu-satunya kelonggaran adalah `'wasm-unsafe-eval'` agar onnxruntime-web dapat mengompilasi WebAssembly; `eval` JavaScript tetap diblokir. Tidak ada permintaan ke pihak ketiga: runtime WASM, model ONNX, font, dan data disajikan dari domain sendiri.
- Berkas terbesar adalah runtime `ort-wasm-simd-threaded.wasm` (sekitar 13,6 MB), di bawah batas 25 MiB per berkas. `npm run build` gagal bila ada berkas di atas batas atau lebih dari 20.000 berkas.
- Model browser dan runtime hanya diunduh ketika pengunjung menekan "Hitung ulang di browser".
- Situs statis tidak tertidur saat sepi.
- Data website dibuat oleh `defect-lab export-web` dan di-commit. Mengubah model berarti menjalankan ulang evaluasi, laporan, ekspor ONNX, dan ekspor web, lalu commit.

## Pemeriksaan setelah merge

1. Dashboard Cloudflare, Workers and Pages, `visual-defect-lab`, Deployments: deployment Production teratas memakai commit terbaru `main` dan berstatus Success.
2. Buka `https://visual-defect-lab.pages.dev/` di jendela penyamaran, buka kelima menu: Inspeksi, Galeri, Kinerja, Kesalahan, Metode.
3. Di Inspeksi, geser pembanding, perbesar foto, lalu tekan "Hitung ulang di browser". Skor langsung harus hampir sama dengan skor yang dihitung Python.
4. Ulangi di ponsel dan coba ganti tema.
