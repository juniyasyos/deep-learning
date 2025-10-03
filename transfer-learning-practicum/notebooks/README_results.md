# Hasil Notebook 01_transfer_learning_intro

Notebook `01_transfer_learning_intro.ipynb` menampilkan contoh alur transfer learning menggunakan dataset dummy. Berikut ringkasan hasil utama yang dapat dijadikan referensi ketika menjalankan notebook.

## Lingkungan & Konfigurasi
- Root proyek: `transfer-learning-practicum`
- Python 3.12.x, PyTorch 2.0.x (contoh offline)
- Perangkat: CPU (demo)
- File konfigurasi: `configs/training.yaml`
  - `num_epochs_feature_extraction`: 1
  - `num_epochs_fine_tuning`: 1
  - `freeze_until`: `all`
  - Optimizer: Adam dengan `lr_feature_extraction=1e-3`, `lr_fine_tuning=1e-4`

## Ringkasan Training
- Tahap feature extraction berjalan 1 epoch dengan metrik contoh:
  - Train loss ≈ 0.692, train acc ≈ 52%
  - Val loss ≈ 0.701, val acc ≈ 48%
- Tahap fine-tuning (kebijakan `freeze_until=all` pada demo dummy) berjalan 1 epoch dengan metrik contoh:
  - Train loss ≈ 0.645, train acc ≈ 58%
  - Val loss ≈ 0.672, val acc ≈ 55%

> Angka di atas dihasilkan dari dataset dummy acak; hasil aktual akan berubah sesuai dataset dan konfigurasi.

## Visualisasi & Artefak
Notebook menyertakan plot dua panel:
1. **Loss per Stage** — membandingkan train vs validation loss untuk setiap tahap.
2. **Accuracy per Stage** — memperlihatkan perkembangan akurasi (0–1) selama dua tahap training.

Artefak contoh yang dihasilkan (akan dibuat ulang saat notebook dijalankan):
- Gambar: `outputs/figures/loss_accuracy_demo.png`
- Ringkasan JSON: `outputs/reports/run_summary.json`
- Checkpoint dummy: `models/demo_model_state_dict.pt`

## Cara Replikasi
1. Pastikan dependensi terpasang (`pip install -r requirements.txt`).
2. Jalankan notebook dari awal hingga akhir untuk menghasilkan hasil aktual sesuai lingkunganmu.
3. Periksa folder `outputs/` dan `models/` untuk artefak training.

## Catatan
- Notebook menggunakan dataset dummy agar aman dieksekusi tanpa sumber data eksternal.
- Jika kamu memiliki GPU, biarkan `device` pada `training.yaml` bernilai `cuda_if_available` dan pastikan CUDA tersedia.
- Untuk demo studi kasus nyata, ganti `SimpleImageDataModule` dengan loader dataset aslimu (mis. Pascal VOC2007) dan sesuaikan `NUM_CLASSES`.

