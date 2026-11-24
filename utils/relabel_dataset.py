"""Siapkan dataset WasteVision (6 kelas) dari dataset/merged_dataset.

Dataset asli adalah gabungan dua sumber yang SEMUA labelnya berkelas 0:
  - garb_*   : foto per material (cardboard/glass/metal/paper/plastic/trash)
  - litter_* : foto sampah liar dengan kotak anotasi asli (tanpa info kelas)

Skrip ini membuat dataset baru di dataset/wastevision_data dengan aturan:
  garb_*cardboard* -> 0 CARDBOARD     garb_*plastic* -> 4 PLASTIC
  garb_*glass*     -> 1 GLASS         garb_*trash*   -> 5 LITTER
  garb_*metal*     -> 2 METAL         litter_*       -> 5 LITTER
  garb_*paper*     -> 3 PAPER

Dataset asli TIDAK diubah.

Pemakaian:
    python utils/relabel_dataset.py
    python utils/relabel_dataset.py --src dataset/merged_dataset --dst dataset/wastevision_data
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLASSES = ["CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC", "LITTER"]
LITTER_ID = 5
KEYWORDS = {
    "cardboard": 0,
    "glass": 1,
    "metal": 2,
    "paper": 3,
    "plastic": 4,
    "trash": LITTER_ID,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def class_from_name(filename_stem: str):
    """Tentukan ID kelas dari nama file; None jika tidak dikenali."""
    # Buang akhiran hash Roboflow (".rf.<hash>") agar tidak ikut dicocokkan
    name = filename_stem.lower().split(".rf.")[0]
    if name.startswith("litter_"):
        return LITTER_ID
    if name.startswith("garb_"):
        for keyword, cid in KEYWORDS.items():
            if keyword in name:
                return cid
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Relabel merged_dataset -> 6 kelas WasteVision")
    p.add_argument("--src", default="dataset/merged_dataset")
    p.add_argument("--dst", default="dataset/wastevision_data")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_absolute():
        src = ROOT / src
    if not dst.is_absolute():
        dst = ROOT / dst

    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        print(f"[ERROR] Folder sumber tidak valid: {src}")
        print("        Harus berisi images/ dan labels/. Lihat dataset/PLACE_DATASET_HERE.txt")
        sys.exit(1)

    if dst.exists():
        print(f"[INFO] Menghapus hasil lama: {dst}")
        shutil.rmtree(dst)

    instances = {s: Counter() for s in SPLITS}
    images_per_split = Counter()
    skipped_unknown = Counter()
    skipped_nolabel = 0
    skipped_empty = 0

    for split in SPLITS:
        img_dir = src / "images" / split
        lbl_dir = src / "labels" / split
        if not img_dir.is_dir():
            print(f"[WARN] Split '{split}' tidak ada, dilewati.")
            continue

        out_img = dst / "images" / split
        out_lbl = dst / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue

            cid = class_from_name(img.stem)
            if cid is None:
                prefix = img.stem.split("_")[0]
                skipped_unknown[prefix] += 1
                continue

            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.is_file():
                skipped_nolabel += 1
                continue

            new_lines = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                parts[0] = str(cid)
                new_lines.append(" ".join(parts))
            if not new_lines:
                skipped_empty += 1
                continue

            (out_lbl / lbl.name).write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            shutil.copy2(img, out_img / img.name)
            images_per_split[split] += 1
            instances[split][cid] += len(new_lines)

    total_images = sum(images_per_split.values())
    if total_images == 0:
        print("[ERROR] Tidak ada gambar yang berhasil dikonversi.")
        sys.exit(1)

    print("\n" + "=" * 62)
    print(" HASIL RELABEL")
    print("=" * 62)
    print(f" Output: {dst}\n")
    print(f" {'Kelas':<12}" + "".join(f"{s:>10}" for s in SPLITS))
    for cid, name in enumerate(CLASSES):
        print(f" {cid} {name:<10}" + "".join(f"{instances[s][cid]:>10}" for s in SPLITS))
    print(f" {'Gambar':<12}" + "".join(f"{images_per_split[s]:>10}" for s in SPLITS))
    print("=" * 62)

    if skipped_unknown:
        print(f"[WARN] Dilewati (awalan nama tidak dikenali): {dict(skipped_unknown)}")
    if skipped_nolabel:
        print(f"[WARN] Dilewati (file label tidak ada): {skipped_nolabel}")
    if skipped_empty:
        print(f"[WARN] Dilewati (label kosong): {skipped_empty}")
    print("\nSelanjutnya jalankan:  python train.py")


if __name__ == "__main__":
    main()
