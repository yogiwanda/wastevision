"""WasteVision - Sistem Deteksi Sampah Liar berbasis YOLOv8 (Streamlit)."""

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from utils.detector import CLASS_NAMES, WasteDetector

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

st.set_page_config(
    page_title="WasteVision",
    page_icon="♻️",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_detector(model_path: str) -> WasteDetector:
    """Muat model sekali saja dan simpan di cache."""
    return WasteDetector(model_path)


# ---------------------------------------------------------------- Header
st.title("♻️ WasteVision")
st.subheader("Sistem Deteksi Sampah Liar Berbasis YOLOv8")
st.write(
    "Unggah foto lingkungan, lalu WasteVision akan mendeteksi dan mengklasifikasikan "
    "sampah ke dalam 6 kategori: **Cardboard, Glass, Metal, Paper, Plastic,** "
    "dan **Litter** (sampah liar / umum)."
)

# ---------------------------------------------------------------- Muat model
detector = None
try:
    detector = load_detector(str(MODEL_PATH))
except FileNotFoundError:
    pass
except Exception as exc:  # model rusak / versi tidak cocok
    st.error(f"Gagal memuat model: {exc}")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    conf_threshold = st.slider(
        "Confidence threshold", min_value=0.1, max_value=0.9, value=0.25, step=0.05,
        help="Semakin tinggi, semakin sedikit tapi semakin yakin deteksinya.",
    )
    iou_threshold = st.slider(
        "IoU threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05,
        help="Ambang NMS untuk menghapus kotak yang tumpang tindih.",
    )

    st.divider()
    st.header("🗂️ Kelas yang Dideteksi")
    class_names = detector.class_names if detector else CLASS_NAMES
    for idx, name in class_names.items():
        st.markdown(f"- `{idx}` {name}")

    st.divider()
    st.header("ℹ️ Info Model")
    if detector:
        st.write(f"**File:** `{MODEL_PATH.name}`")
        st.write(f"**Jumlah kelas:** {detector.num_classes}")
    else:
        st.warning("Model belum dimuat.")

# ---------------------------------------------------------------- Model belum ada
if detector is None:
    if not MODEL_PATH.is_file():
        st.error(
            "Model belum tersedia. Latih model terlebih dahulu dengan `python train.py`"
        )
        st.info(
            "Setelah training selesai, file `best.pt` akan otomatis tersimpan di "
            "folder `models/`. Muat ulang halaman ini setelahnya."
        )
    st.stop()

# ---------------------------------------------------------------- Upload
uploaded = st.file_uploader(
    "Unggah gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("👆 Unggah gambar untuk memulai deteksi.")
    st.stop()

try:
    image = Image.open(uploaded)
    image = ImageOps.exif_transpose(image).convert("RGB")
except Exception:
    st.error("File gambar tidak dapat dibaca. Coba unggah file JPG/PNG lain.")
    st.stop()

# ---------------------------------------------------------------- Deteksi
with st.spinner("Mendeteksi sampah..."):
    results = detector.detect(image, conf=conf_threshold, iou=iou_threshold)
    annotated = detector.get_annotated_image(results)
    summary = detector.summarize(results)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("#### Gambar Asli")
    st.image(image, use_container_width=True)
with col_right:
    st.markdown("#### Hasil Deteksi")
    st.image(annotated, use_container_width=True)

# ---------------------------------------------------------------- Hasil
st.divider()
st.subheader("📊 Ringkasan Hasil")

m1, m2, m3 = st.columns(3)
m1.metric("Total objek terdeteksi", summary["total"])
m2.metric("Jumlah kelas unik", summary["unique_classes"])
m3.metric("Rata-rata confidence", f"{summary['avg_confidence'] * 100:.1f}%")

if summary["total"] == 0:
    st.warning(
        "Tidak ada sampah terdeteksi. Coba turunkan *confidence threshold* di sidebar."
    )
else:
    df = pd.DataFrame(
        {
            "Kelas": list(summary["per_class"].keys()),
            "Jumlah": list(summary["per_class"].values()),
        }
    )

    t_col, c_col = st.columns(2)
    with t_col:
        st.markdown("#### Rincian per Kelas")
        st.dataframe(df, use_container_width=True, hide_index=True)
    with c_col:
        st.markdown("#### Jumlah per Kelas")
        st.bar_chart(df.set_index("Kelas")["Jumlah"])

st.caption("WasteVision • YOLOv8 + Streamlit • Proyek Computer Vision")
