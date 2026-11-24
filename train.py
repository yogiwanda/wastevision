"""Script training YOLOv8 untuk WasteVision.

Contoh:
    python train.py
    python train.py --epochs 100 --model yolov8s.pt --batch 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
FINAL_MODEL = MODELS_DIR / "best.pt"
RESOLVED_YAML = ROOT / "data" / ".data_resolved.yaml"

DATASET_HELP = """
============================================================
 DATASET BELUM SIAP
============================================================
1. Unduh dataset dari Kaggle dan letakkan folder 'merged_dataset'
   di dalam 'dataset/':
     https://www.kaggle.com/datasets/pravesh212006/garbage-detection-dataset

2. Siapkan label 6 kelas (sekali saja):
     python utils/relabel_dataset.py
   -> membuat dataset/wastevision_data/ (images/ dan labels/ untuk
      train, val, test)

3. Jalankan kembali:  python train.py

Detail lengkap ada di dataset/PLACE_DATASET_HERE.txt
============================================================
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training YOLOv8 untuk WasteVision")
    p.add_argument("--epochs", type=int, default=50, help="Jumlah epoch (default: 50)")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Bobot awal (default: yolov8n.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="Ukuran gambar (default: 640)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--data", type=str, default="data/data.yaml", help="Path data.yaml (default: data/data.yaml)")
    return p.parse_args()


def has_files(folder: Path) -> bool:
    return folder.is_dir() and any(folder.iterdir())


def load_config(data_path: Path) -> dict:
    """Baca data.yaml dan ubah 'path' menjadi absolut.

    Ultralytics tidak menghitung path relatif terhadap lokasi file yaml,
    jadi path dihitung di sini relatif terhadap lokasi data.yaml.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg.get("path", "../dataset"))
    if not base.is_absolute():
        base = (data_path.parent / base).resolve()
    cfg["path"] = str(base)
    return cfg


def validate_inputs(data_path: Path) -> dict:
    """Pastikan data.yaml dan dataset tersedia sebelum training."""
    if not data_path.is_file():
        print(f"[ERROR] File konfigurasi tidak ditemukan: {data_path}")
        sys.exit(1)

    cfg = load_config(data_path)
    base = Path(cfg["path"])
    missing = [
        str(base / cfg[key])
        for key in ("train", "val")
        if key not in cfg or not has_files(base / cfg[key])
    ]
    if not DATASET_DIR.is_dir() or missing:
        print(DATASET_HELP)
        if missing:
            print("Folder berikut kosong / tidak ditemukan:")
            for m in missing:
                print(f"  - {m}")
        sys.exit(1)
    return cfg


def check_labels(cfg: dict) -> None:
    """Pastikan label train benar-benar memuat beberapa kelas.

    Mencegah training sia-sia ketika semua label berkelas sama
    (mis. dataset belum direlabel dengan utils/relabel_dataset.py).
    """
    base = Path(cfg["path"])
    lbl_dir = base / str(cfg["train"]).replace("images", "labels", 1)
    if not lbl_dir.is_dir():
        print(f"[WARN] Folder label tidak ditemukan, pengecekan kelas dilewati: {lbl_dir}")
        return

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    nc = int(cfg.get("nc", len(names)))

    counts: dict = {}
    for f in lbl_dir.glob("*.txt"):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if parts:
                counts[parts[0]] = counts.get(parts[0], 0) + 1

    print("[INFO] Distribusi kelas pada label train:")
    for cid in sorted(counts, key=lambda x: int(x) if x.isdigit() else 999):
        label = names.get(int(cid), "?") if cid.isdigit() else "?"
        print(f"         {cid:>2} {label:<14} {counts[cid]:>7}")

    bad = [c for c in counts if not c.isdigit() or int(c) >= nc]
    if bad:
        print(f"\n[ERROR] Ada ID kelas di luar rentang 0..{nc - 1}: {sorted(bad)}")
        sys.exit(1)

    if nc > 1 and len(counts) < 2:
        only = next(iter(counts), "-")
        print(
            f"\n[ERROR] Semua label berkelas sama (ID {only}), padahal nc={nc}.\n"
            "        Model tidak akan bisa membedakan kelas. Jalankan dulu:\n"
            "            python utils/relabel_dataset.py\n"
            "        lalu pastikan data/data.yaml memakai path: ../dataset/wastevision_data\n"
        )
        sys.exit(1)

    missing = [str(i) for i in range(nc) if str(i) not in counts]
    if missing:
        print(f"[WARN] Kelas tanpa contoh di label train: {missing}")
    print()


def build_resolved_yaml(cfg: dict) -> Path:
    """Tulis salinan data.yaml dengan 'path' absolut untuk dipakai Ultralytics."""
    cfg = dict(cfg)
    base = Path(cfg["path"])

    # Hindari error jika split test tidak ada
    if "test" in cfg and not has_files(base / cfg["test"]):
        cfg.pop("test")

    RESOLVED_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(RESOLVED_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return RESOLVED_YAML


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    cfg = validate_inputs(data_path)
    check_labels(cfg)
    resolved = build_resolved_yaml(cfg)

    from ultralytics import YOLO

    print(f"\n[INFO] Model awal : {args.model}")
    print(f"[INFO] Dataset    : {cfg['path']}")
    print(f"[INFO] Epochs     : {args.epochs} | imgsz: {args.imgsz} | batch: {args.batch}\n")

    model = YOLO(args.model)
    model.train(
        data=str(resolved),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs"),
        name="wastevision",
    )

    # Lokasi weights terbaik dari run ini
    best_src = Path(model.trainer.best)
    if not best_src.is_file():
        best_src = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_src.is_file():
        print("[ERROR] best.pt tidak ditemukan setelah training.")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_src, FINAL_MODEL)
    print(f"\n[OK] Weights terbaik disalin ke: {FINAL_MODEL}")

    # Evaluasi akhir pada set validasi
    metrics = YOLO(str(FINAL_MODEL)).val(data=str(resolved), split="val", verbose=False)
    print("\n" + "=" * 50)
    print(" RINGKASAN HASIL TRAINING (validation set)")
    print("=" * 50)
    print(f" mAP@0.5      : {metrics.box.map50:.4f}")
    print(f" mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f" Precision    : {metrics.box.mp:.4f}")
    print(f" Recall       : {metrics.box.mr:.4f}")
    print("=" * 50)
    print("\nJalankan aplikasi dengan:  streamlit run app.py")


if __name__ == "__main__":
    main()
