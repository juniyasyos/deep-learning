# Hasil Notebook 01_transfer_learning_intro

Notebook `01_transfer_learning_intro.ipynb` kini mencontohkan alur transfer learning dengan dataset nyata (default: CIFAR-10)
serta opsi memakai `ImageFolder` pribadi. Gunakan ringkasan berikut sebagai panduan ketika menjalankan notebook.

## Lingkungan & Konfigurasi
- Root proyek: `transfer-learning-practicum`
- File konfigurasi: `configs/training.yaml`
  - `dataset_name`: `cifar10` (ubah ke `imagefolder` jika memakai data sendiri)
  - `data_root`: `data/raw`
  - `train_subset_fraction`: 0.05 (sekitar ≤1K sampel agar cepat)
  - `val_subset_fraction`: 0.1
  - `max_train_samples`: 1024
  - `max_val_samples`: 512
  - `num_epochs_feature_extraction`: 2
  - `num_epochs_fine_tuning`: 2
  - `freeze_until`: `layer4`
  - Optimizer Adam (`lr_feature_extraction=1e-3`, `lr_fine_tuning=1e-4`, `weight_decay=1e-4`)

## Ekspektasi Hasil
- Saat `dataset_name=cifar10` dan subset 20%, kamu akan melihat 4 epoch total (2 freeze + 2 fine-tune).
- Loss dan akurasi akan bergantung pada subset yang diambil; catat tren penurunan loss dan kenaikan akurasi antara tahap freeze dan fine-tune.
- Artefak otomatis:
  - Plot dua panel (loss & akurasi) disimpan ke `outputs/figures/loss_accuracy_demo.png`.
  - Ringkasan konfigurasi + metrik ke `outputs/reports/run_summary.json`.
  - Checkpoint hasil fine-tune ke `models/demo_model_state_dict.pt`.

> Jalankan notebook untuk menghasilkan nilai metrik aktual. Berkas artefak akan ditimpa setiap kali eksperimen baru dijalankan.

## Cara Menjalankan
1. Instal dependensi (Torch + Torchvision) pada environment yang sudah ada: `pip install --user -r requirements.txt`.
2. Eksekusi notebook dari awal. Pada eksekusi pertama, CIFAR-10 akan terunduh otomatis ke `data/raw`. Tanpa internet, siapkan folder `train/` dan `val/` lalu set `dataset_name=imagefolder`.
3. Setelah training selesai, bukalah plot dan ringkasan untuk refleksi hasil. Dokumentasikan perubahan konfigurasi pada catatan praktikummu.

## Tips Lanjutan
- Tingkatkan `train_subset_fraction` secara bertahap bila ingin mendekati hasil penuh CIFAR-10.
- Set `freeze_until=none` untuk full fine-tuning setelah kamu puas dengan hasil parsial.
- Ganti dataset ke `ImageFolder` medis atau domainmu sendiri untuk mempraktikkan transfer learning di kasus nyata.

