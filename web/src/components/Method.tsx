import { bytes } from "../lib/format.ts";
import { CLASS_NAMES } from "../lib/labels.ts";
import type { Summary } from "../lib/types.ts";

export function Method({ summary }: { summary: Summary }) {
  const d = summary.dataset;
  const c = d.counts;
  return (
    <div className="page prose">
      <div className="page-head">
        <h1>Cara kerja dan batasan</h1>
        <p>
          Proyek ini menjawab satu pertanyaan: bila sebuah model hanya pernah melihat produk yang bagus, seberapa baik ia menemukan dan menunjuk letak
          cacat pada produk baru? Semua angka dapat dihitung ulang dari kode di repositori.
        </p>
      </div>

      <section className="block">
        <h2>Data</h2>
        <p>
          Dataset {d.name} ({d.version}) dari Amazon, 12 produk: {Object.values(CLASS_NAMES).join(", ")}. Pembagian resmi VisA dipakai apa adanya. Dari foto
          latih resmi (semuanya normal), 10% disisihkan secara acak dengan benih tetap sebagai validasi. Jadi ada {c.fit} foto normal untuk belajar, {c.val} foto
          normal validasi untuk menentukan ambang, dan {c.test_normal + c.test_defect} foto uji ({c.test_defect} cacat dengan masker area cacat,{" "}
          {c.test_normal} normal). Semua foto diperkecil ke 256 x 256 piksel tanpa dipotong.
        </p>
        <p className="muted small">
          Arsip diverifikasi dengan SHA-256 <span className="mono break">{d.sha256}</span>. Lisensi {d.licence}. Sitasi: {d.citation}
        </p>
      </section>

      <section className="block">
        <h2>Tiga model</h2>
        <p>
          Ketiganya memakai cara menilai yang sama: foto dipecah menjadi 32 x 32 kotak kecil (patch), setiap patch diubah menjadi deretan angka, lalu
          dibandingkan dengan patch paling mirip dari foto normal. Makin jauh jaraknya, makin tinggi skornya. Skor patch diperhalus menjadi peta skor, dan skor
          foto adalah titik tertinggi di peta itu. Perbedaannya hanya pada cara mengubah patch menjadi angka.
        </p>
        <ul>
          <li>
            <strong>Klasik (baseline).</strong> Tanpa deep learning. Dengan OpenCV dihitung rata-rata dan sebaran warna, arah gradien tepi, dan pola tekstur
            Local Binary Pattern, total 24 angka per patch.
          </li>
          <li>
            <strong>ResNet18 dan WideResNet50 (transfer learning).</strong> Lapisan tengah jaringan yang sudah dilatih pada ImageNet dipakai sebagai pengenal
            ciri tanpa dilatih ulang (metode PatchCore). Memori patch normal dirangkum dengan pemilihan coreset agar ringan.
          </li>
        </ul>
        <p>
          Ambang operasi tiap produk adalah persentil ke-95 skor foto normal validasi, sehingga sekitar 5% foto normal diperkirakan ditolak. Ambang yang sama
          menentukan area yang ditandai. Foto uji tidak dipakai untuk memilih ambang, pengaturan, atau model.
        </p>
      </section>

      <section className="block">
        <h2>Model di browser</h2>
        {summary.onnx ? (
          <p>
            Tombol "Hitung ulang di browser" menjalankan ResNet18 yang diekspor ke ONNX dan dipadatkan ke int8 ({bytes(summary.onnx.backbone.bytes)}) dengan
            onnxruntime-web (WebAssembly). Foto contoh disimpan sebagai PNG 256 x 256 tanpa kompresi lossy agar piksel yang masuk ke model sama persis dengan
            di Python. Skor yang dihitung browser ditampilkan berdampingan dengan skor tersimpan agar bisa dicek. Hasil WideResNet50 dan model klasik
            dihitung lebih dulu di Python dan ditandai sebagai skor tersimpan.
          </p>
        ) : (
          <p>Model browser belum diekspor. Semua peta skor di website ini dihitung lebih dulu di Python.</p>
        )}
      </section>

      <section className="block">
        <h2>Batasan</h2>
        <ul>
          <li>Resolusi 256 x 256 membuat cacat yang sangat kecil (di bawah 0,2% luas foto) sulit ditandai. Lihat halaman Kesalahan.</li>
          <li>Foto dipaksa menjadi persegi. Produk tampak sedikit memipih, dan peta skor ditarik kembali ke rasio aslinya saat ditampilkan.</li>
          <li>Ambang dipilih dari 45 sampai 90 foto normal validasi per produk. Tingkat alarm palsu pada data uji bisa berbeda dari 5%.</li>
          <li>Hanya 50 sampai 101 foto normal dan 100 foto cacat uji per produk, sehingga selang kepercayaan per produk lebar.</li>
          <li>Model mengenal produk dari foto latihnya saja. Kamera, pencahayaan, atau produk lain membutuhkan foto normal baru.</li>
          <li>Ini demo portofolio dengan data publik, bukan sistem inspeksi yang tervalidasi untuk produksi.</li>
        </ul>
      </section>

      <section className="block">
        <h2>Status pengerjaan</h2>
        <ul>
          <li>Fitting dan evaluasi dijalankan di CPU (4 inti) pada lingkungan pengembangan, bukan di Kaggle. Notebook Kaggle tersedia di repositori tetapi belum dijalankan di Kaggle.</li>
          <li>PatchCore tidak melatih ulang bobot jaringan; tidak ada fine-tuning.</li>
        </ul>
      </section>
    </div>
  );
}
