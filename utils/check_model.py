"""Uji cepat apakah model benar-benar membedakan kelas.

Untuk tiap kelompok gambar (nama file garb_*cardboard*, garb_*glass*, dst. dan
litter_*), skrip menjalankan model dan menghitung seberapa sering kelas
dengan confidence tertinggi sesuai dengan kelas yang diharapkan.

Pemakaian:
    python utils/check_model.py
    python utils/check_model.py --model models/best.pt --images dataset/wastevision_data/images/test --n 40

Cara membaca hasil:
  - Model 6 kelas yang sehat: tiap baris punya "benar" yang tinggi
    dan sebaran prediksi terpusat pada kelas yang sesuai.
  - Model yang hanya belajar 1 kelas: hampir semua prediksi jatuh
    ke satu nama kelas (mis. LITTER) pada SEMUA baris.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEYWORDS = {
    "cardboard": "CARDBOARD",
    "glass": "GLASS",
    "metal": "METAL",
    "paper": "PAPER",
    "plastic": "PLASTIC",
    "trash": "LITTER",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def group_of(stem: str):
    """Kelas yang diharapkan berdasarkan nama file (None jika tidak dikenali)."""
    name = stem.lower().split(".rf.")[0]
    if name.startswith("litter_"):
        return "LITTER"
    if name.startswith("garb_"):
        for key, label in KEYWORDS.items():
            if key in name:
                return label
    return None


def find_images_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else ROOT / p
    for cand in ("dataset/wastevision_data/images/test", "dataset/merged_dataset/images/test",
                 "dataset/wastevision_data/images/val", "dataset/merged_dataset/images/val"):
        if (ROOT / cand).is_dir():
            return ROOT / cand
    return ROOT / "dataset/merged_dataset/images/test"


def main() -> None:
    ap = argparse.ArgumentParser(description="Uji kemampuan model membedakan kelas")
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--images", default=None, help="Folder gambar uji (default: otomatis)")
    ap.add_argument("--n", type=int, default=30, help="Maks gambar per kelompok (default: 30)")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    img_dir = find_images_dir(args.images)

    if not model_path.is_file():
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        sys.exit(1)
    if not img_dir.is_dir():
        print(f"[ERROR] Folder gambar tidak ditemukan: {img_dir}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = dict(model.names)
    print(f"[INFO] Model : {model_path.name} | kelas: {names}")
    print(f"[INFO] Gambar: {img_dir}\n")

    groups: dict[str, list[Path]] = {}
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() in IMG_EXT:
            g = group_of(f.stem)
            if g:
                groups.setdefault(g, []).append(f)
    if not groups:
        print("[ERROR] Tidak ada gambar dengan awalan garb_/litter_ di folder tersebut.")
        sys.exit(1)

    print(f"{'Diharapkan':<12}{'Gambar':>7}{'Benar':>8}   Sebaran prediksi (kelas dengan confidence tertinggi)")
    print("-" * 78)
    all_pred, total, total_ok = Counter(), 0, 0
    for label in ("CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "LITTER"):
        files = groups.get(label, [])[: args.n]
        if not files:
            continue
        preds = Counter()
        for f in files:
            res = model.predict(str(f), conf=args.conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                preds["(tidak ada)"] += 1
            else:
                i = int(res.boxes.conf.argmax())
                preds[names[int(res.boxes.cls[i])]] += 1
        ok = preds.get(label, 0)
        total += len(files)
        total_ok += ok
        all_pred.update(preds)
        dist = ", ".join(f"{k}:{v}" for k, v in preds.most_common())
        print(f"{label:<12}{len(files):>7}{ok / len(files) * 100:>7.0f}%   {dist}")

    print("-" * 78)
    print(f"Akurasi total (top-1): {total_ok / total * 100:.0f}% dari {total} gambar")
    top, cnt = all_pred.most_common(1)[0]
    if cnt / total > 0.8:
        print(f"\n[PERINGATAN] {cnt / total * 100:.0f}% prediksi jatuh ke '{top}'. "
              "Model kemungkinan tidak membedakan kelas (label training bermasalah).")
    else:
        print("\n[OK] Prediksi tersebar ke beberapa kelas, model tampak membedakan kelas.")


if __name__ == "__main__":
    main()
