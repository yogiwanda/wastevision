# ♻️ WasteVision: Sistem Deteksi Sampah Berbasis YOLOv8

WasteVision adalah aplikasi web *Computer Vision* untuk mendeteksi dan mengklasifikasikan
sampah liar pada gambar menggunakan **YOLOv8** (Ultralytics) dengan antarmuka **Streamlit**.

## 🎯 Fitur
- Deteksi 6 kelas sampah: `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`, `LITTER`
- Upload gambar JPG/JPEG/PNG, tampilan **gambar asli vs hasil deteksi** berdampingan
- Pengaturan *confidence* dan *IoU threshold* secara interaktif
- Ringkasan: total objek, jumlah kelas unik, rata-rata confidence, tabel & grafik per kelas
- Script training otomatis yang menyalin `best.pt` ke `models/`

## 📁 Struktur Proyek
```
wastevision/
├── app.py                     # Aplikasi Streamlit
├── train.py                   # Script training YOLOv8
├── requirements.txt           # Dependensi Python
├── packages.txt               # Dependensi sistem (Streamlit Cloud)
├── README.md
├── .gitignore
├── .streamlit/config.toml     # Tema Streamlit
├── data/data.yaml             # Konfigurasi dataset YOLO
├── models/                    # Tempat best.pt
├── dataset/                   # Tempat dataset (tidak di-commit)
├── utils/detector.py          # Kelas WasteDetector
├── utils/relabel_dataset.py   # Menyiapkan label 6 kelas dari dataset Kaggle
├── utils/check_model.py       # Uji apakah model membedakan 6 kelas
└── assets/                    # Screenshot (opsional)
```

## 📦 Dataset
Sumber: [Garbage Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/pravesh212006/garbage-detection-dataset)

Folder `merged_dataset` dari Kaggle adalah **gabungan dua sumber** yang semua labelnya
berkelas `0`:

| Sumber (awalan nama file) | Isi | Bentuk label |
|---|---|---|
| `garb_*` (cardboard, glass, metal, paper, plastic, trash) | Foto per material | Kotak seukuran seluruh gambar |
| `litter_*` | Foto sampah liar di lingkungan | Kotak anotasi asli, bisa banyak per gambar |

Karena itu label perlu dipetakan ulang dari nama file:

| ID | Kelas | Asal |
|---|---|---|
| 0 | CARDBOARD | `garb_*cardboard*` |
| 1 | GLASS | `garb_*glass*` |
| 2 | METAL | `garb_*metal*` |
| 3 | PAPER | `garb_*paper*` |
| 4 | PLASTIC | `garb_*plastic*` |
| 5 | LITTER | `garb_*trash*` dan seluruh `litter_*` |

**Langkah:**
1. Unduh dataset dari Kaggle, lalu letakkan folder `merged_dataset` di dalam `dataset/`:
   ```
   dataset/merged_dataset/
   ├── images/   (train/, val/, test/)
   └── labels/   (train/, val/, test/)
   ```
2. Jalankan sekali:
   ```bash
   python utils/relabel_dataset.py
   ```
   Ini membuat `dataset/wastevision_data/` (dataset asli tidak diubah) dan menampilkan
   jumlah objek per kelas untuk tiap split.

> **Keterbatasan (cantumkan di laporan):** kotak anotasi `garb_*` menutupi seluruh gambar,
> sedangkan kotak `litter_*` tepat di objek. Akibatnya model cenderung menghasilkan kotak
> besar untuk kelas material dan kotak yang lebih presisi untuk `LITTER`. Kelas
> `BIODEGRADABLE` tidak tersedia pada dataset ini. `LITTER` mencakup sampah umum yang
> tidak termasuk lima material lainnya.

## 🚀 Instalasi Lokal
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🏋️ Training Model
```bash
python train.py
```
Opsi tambahan:
```bash
python train.py --epochs 100 --model yolov8s.pt --imgsz 640 --batch 16
```

| Argumen    | Default            | Keterangan               |
|------------|--------------------|--------------------------|
| `--epochs` | 50                 | Jumlah epoch             |
| `--model`  | `yolov8n.pt`       | Bobot awal YOLOv8        |
| `--imgsz`  | 640                | Ukuran gambar            |
| `--batch`  | 16                 | Batch size               |
| `--data`   | `data/data.yaml`   | File konfigurasi dataset |

Setelah selesai, `best.pt` otomatis disalin ke `models/best.pt` dan ringkasan
mAP, precision, dan recall ditampilkan di terminal. Hasil lengkap (grafik, confusion
matrix) ada di `runs/wastevision*/`.

💡 **Tips:** Tidak punya GPU? Jalankan `train.py` di Google Colab / Kaggle Notebook
(aktifkan GPU), lalu unduh `best.pt` dan letakkan di folder `models/`.

## 🔎 Uji Kualitas Model
```bash
python utils/check_model.py
```
Menjalankan model pada gambar uji per kelompok (`garb_*` per material dan `litter_*`) lalu
menampilkan akurasi top-1 dan sebaran prediksinya. Jika hampir semua prediksi jatuh ke satu
kelas, label training kemungkinan bermasalah.

## ▶️ Menjalankan Aplikasi
```bash
streamlit run app.py
```
Buka `http://localhost:8501`.

## ☁️ Deploy ke Streamlit Community Cloud
1. **Pastikan `models/best.pt` ada.** Aplikasi butuh file ini saat berjalan. File ini
   sengaja *tidak* di-ignore oleh `.gitignore` (aturan `!models/best.pt`). Ukuran
   YOLOv8n sekitar 6 MB, jauh di bawah batas GitHub.
2. Push proyek ke repositori GitHub:
   ```bash
   git init
   git add .
   git commit -m "WasteVision"
   git branch -M main
   git remote add origin https://github.com/<username>/wastevision.git
   git push -u origin main
   ```
   Pastikan `git status` menunjukkan `models/best.pt` ikut ter-commit dan
   folder `dataset/` tidak.
3. Buka [share.streamlit.io](https://share.streamlit.io) → **Create app** → pilih repo,
   branch `main`, dan *Main file path* `app.py`.
4. Di **Advanced settings**, pilih Python **3.11** (atau 3.12).
5. Klik **Deploy**.

**Catatan deployment**
- `requirements.txt` memakai PyTorch **CPU** agar instalasi ringan.
- `packages.txt` berisi `libgl1`, yaitu pustaka sistem yang dibutuhkan OpenCV agar tidak
  error `libGL.so.1 not found`.
- Streamlit Cloud tidak memakai GPU, jadi inferensi berjalan di CPU. Model `yolov8n`
  sudah cukup cepat untuk satu gambar.

## 🧩 Penjelasan Kode Singkat
- `utils/detector.py`: kelas `WasteDetector` (`detect`, `get_annotated_image`, `summarize`).
- `app.py`: UI Streamlit, model dimuat sekali dengan `@st.cache_resource`.
- `utils/relabel_dataset.py`: memetakan ulang label dari nama file menjadi 6 kelas.
- `train.py`: validasi dataset → training → salin `best.pt` → tampilkan metrik.

## 🛠️ Pemecahan Masalah
| Masalah | Solusi |
|---|---|
| "Model belum tersedia" | Jalankan `python train.py` atau letakkan `best.pt` di `models/` |
| Training berhenti dan meminta dataset | Pastikan sudah menjalankan `python utils/relabel_dataset.py` dan folder `dataset/wastevision_data/images/train` ada |
| Aplikasi menampilkan kelas yang salah | Hapus `models/best.pt` lama (hasil training sebelum relabel) lalu latih ulang |
| `libGL.so.1` error saat deploy | Pastikan `packages.txt` ada di root repo |
| Memori habis saat training | Kecilkan `--batch` (mis. 8 atau 4) |

## 📚 Teknologi
Python • Ultralytics YOLOv8 • PyTorch • OpenCV • Streamlit • Pandas

## 📄 Lisensi & Kredit
Dataset © pemilik dataset di Kaggle. Proyek ini dibuat untuk keperluan akademik.
