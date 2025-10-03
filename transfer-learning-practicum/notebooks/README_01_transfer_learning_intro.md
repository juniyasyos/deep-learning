# README — Notebook 01_transfer_learning_intro

Notebook `01_transfer_learning_intro.ipynb` memandu kamu mempraktikkan transfer learning berbasis PyTorch dengan contoh
backbone ResNet-18 pretrained ImageNet. Dokumen ini merangkum alur kegiatan di dalam notebook sekaligus meninjau konsep
dasar yang perlu dipahami sebelum mulai mengeksplorasi kode.

## Tujuan Notebook
- Mengenali motivasi transfer learning dan perbandingan strategi freeze vs fine-tune.
- Menjalankan pipeline training pada dataset nyata (default: CIFAR-10 dengan subset ringan) secara end-to-end.
- Mengamati perubahan loss dan akurasi ketika berpindah dari tahap feature extraction ke tahap fine-tuning parsial.
- Menyusun refleksi pribadi melalui checklist pemahaman, ringkasan diskusi, dan rencana eksperimen lanjutan.

## Prasyarat & Persiapan
- Environment Python dengan paket pada `requirements.txt` (Torch, Torchvision, Matplotlib, Pandas, PyYAML, dsb.).
- Koneksi internet saat pertama kali mengunduh CIFAR-10. Jika offline, siapkan dataset lokal berformat `ImageFolder`.
- Pemahaman dasar convolutional neural networks (CNN), supervised learning, dan operasi PyTorch (`DataLoader`, optimizer,
  criterion, forward/backward pass).

## Struktur Kegiatan dalam Notebook
1. **A. Judul & Tujuan Pembelajaran** — menyelaraskan ekspektasi belajar dan indikator keberhasilan.
2. **B. Recall & Icebreaker** — refleksi pengalaman agar konsep transfer learning terasa relevan.
3. **C. Ringkasan Teori** — menggali manfaat, jenis, dan strategi adaptasi (feature extraction vs fine-tuning).
4. **D. Setup Lingkungan** — memeriksa versi dan menyiapkan folder output.
5. **E. Konfigurasi (Code)** — membaca `configs/training.yaml`, termasuk pengaturan dataset, subset, epoch, dan freeze policy.
6. **F. DataModule Real-World (Code)** — memuat CIFAR-10 atau `ImageFolder` kustom dengan transformasi ImageNet standar,
   serta membuat subset agar latihan cepat.
7. **G. Bangun Model Pretrained (Code)** — memisahkan feature extractor dan classifier head pada ResNet-18 pretrained.
8. **H. Loop Train Generic (Code)** — menjalankan dua tahap training: feature extraction (freeze) dan fine-tuning parsial.
9. **I. Plot & Logging (Code)** — merangkum metrik ke DataFrame, memvisualkan loss/akurasi, dan menyimpan artefak.
10. **J. Ringkasan & Diskusi** — memicu refleksi kritis; cocok untuk diskusi kelompok atau laporan singkat.
11. **K. Checklist Pemahaman** — self-check sebelum lanjut ke materi studi kasus.
12. **M. Panduan Langkah Transfer Learning** — urutan praktis menerapkan transfer learning dari ImageNet ke VOC2007 atau dataset lain.
13. **N. Glosarium & Istilah Kunci** — kamus mini istilah penting.

## Alur Eksekusi Singkat
1. **Konfigurasi** — pastikan parameter di `configs/training.yaml` sesuai kebutuhan (dataset, subset fraction, epoch, freeze policy).
2. **Data Loading** — jalankan sel DataModule untuk mengunduh/membaca CIFAR-10 dan menampilkan ukuran batch.
3. **Model Setup** — bangun ResNet-18 pretrained, hitung parameter trainable untuk mode freeze vs fine-tune.
4. **Training Tahap 1 (Feature Extraction)** — latih hanya classifier head dengan backbone beku.
5. **Training Tahap 2 (Fine-Tuning Parsial)** — buka blok `layer4`, latih ulang dengan learning rate lebih kecil.
6. **Evaluasi & Logging** — periksa DataFrame metrik, plot loss/akurasi, dan lihat file `outputs/` serta `models/`.
7. **Refleksi** — jawab pertanyaan diskusi, isi checklist, dan rancang eksperimen lanjutan (mis. ganti dataset, tambah epoch).

## Materi Dasar (Ringkasan Teori)
- **Transfer Learning**: memanfaatkan pengetahuan model pretrained (biasanya dilatih pada dataset besar seperti ImageNet) untuk tugas baru dengan data lebih sedikit.
- **Feature Extraction (Freeze)**: membekukan backbone; hanya melatih classifier baru. Cocok ketika domain target mirip domain sumber.
- **Fine-Tuning Parsial**: membuka beberapa lapisan akhir agar fitur bisa beradaptasi. Seimbang antara kecepatan dan adaptasi.
- **Fine-Tuning Penuh**: melatih seluruh backbone; digunakan jika domain sangat berbeda atau data target cukup besar.
- **Regularisasi Implisit**: bobot pretrained bertindak sebagai titik awal yang stabil, membantu menghindari overfitting di dataset kecil.
- **Kesesuaian Domain**: semakin jauh domain target dari domain sumber, semakin besar kebutuhan untuk membuka lapisan tambahan dan menyesuaikan augmentasi.

## Tips Eksplorasi Lanjutan
- Naikkan `train_subset_fraction` mendekati 1.0 untuk melihat peningkatan performa ketika data lebih banyak.
- Ubah `freeze_until` menjadi `none` untuk fine-tuning penuh setelah tahap parsial.
- Ganti `dataset_name` menjadi `imagefolder` lalu gunakan dataset bidang studi (mis. VOC2007 atau dataset medis).
- Tambahkan callback sederhana (mis. early stopping atau scheduler) untuk memantau performa secara realistis.
- Bandingkan hasil ResNet-18 dengan backbone lain (mis. MobileNetV2) untuk melihat trade-off kecepatan vs akurasi.

## Referensi Pendukung
- Pan, S. J., & Yang, Q. (2010). *A Survey on Transfer Learning*.
- Dokumentasi resmi PyTorch: Transfer Learning Tutorial.
- Goodfellow, Bengio, & Courville. *Deep Learning* — bab mengenai representasi dan transfer.

Gunakan README ini sebagai panduan cepat saat menjelajah notebook. Catat eksperimenmu beserta konfigurasi agar proses
belajar terstruktur dan mudah direplikasi.
