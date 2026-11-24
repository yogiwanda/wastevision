"""Wrapper model YOLOv8 untuk deteksi sampah liar (WasteVision)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from PIL import Image

# Nama kelas bawaan (urutan sesuai data/data.yaml)
CLASS_NAMES = {
    0: "CARDBOARD",
    1: "GLASS",
    2: "METAL",
    3: "PAPER",
    4: "PLASTIC",
    5: "LITTER",
}

ImageInput = Union[Image.Image, np.ndarray, str, Path]


class WasteDetector:
    """Pembungkus (wrapper) model YOLOv8 hasil training."""

    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model tidak ditemukan di '{self.model_path}'. "
                "Latih model terlebih dahulu dengan `python train.py`, "
                "atau letakkan file best.pt di folder models/."
            )

        # Import di sini agar pesan error model hilang lebih cepat & jelas
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_path))
        self.class_names: Dict[int, str] = dict(self.model.names)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def detect(self, image: ImageInput, conf: float = 0.25, iou: float = 0.45):
        """Jalankan deteksi.

        Args:
            image: PIL.Image, path file, atau numpy array berformat RGB.
            conf: ambang confidence.
            iou: ambang IoU untuk NMS.

        Returns:
            List hasil `ultralytics.engine.results.Results`.
        """
        if isinstance(image, np.ndarray):
            # Ultralytics menganggap ndarray sebagai BGR -> konversi dari RGB
            image = image[..., ::-1].copy()
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        return self.model.predict(source=image, conf=conf, iou=iou, verbose=False)

    @staticmethod
    def get_annotated_image(results) -> np.ndarray:
        """Kembalikan gambar beranotasi sebagai array numpy RGB."""
        bgr = results[0].plot()
        return bgr[..., ::-1].copy()

    @staticmethod
    def summarize(results) -> Dict[str, Any]:
        """Ringkas hasil deteksi.

        Returns:
            dict dengan kunci:
              - total: jumlah objek terdeteksi
              - unique_classes: jumlah kelas unik
              - per_class: {nama_kelas: jumlah} (urut dari terbanyak)
              - avg_confidence: rata-rata confidence (0-1)
        """
        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return {
                "total": 0,
                "unique_classes": 0,
                "per_class": {},
                "avg_confidence": 0.0,
            }

        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        names = result.names

        counter = Counter(names[i] for i in class_ids)
        per_class = dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

        return {
            "total": int(len(class_ids)),
            "unique_classes": len(per_class),
            "per_class": per_class,
            "avg_confidence": float(confidences.mean()),
        }
