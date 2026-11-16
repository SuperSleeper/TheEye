"""Recognise enrolled people (family members) by face.

YuNet finds faces and SFace turns each into a 128-number fingerprint (both small OpenCV models, run on the CPU,
files in <install>/models). For every person TheEye sees, the face is taken from the full-resolution frame of the
track's best moment and stored in the faces table. A face close enough to someone's confirmed faces is tagged
with their name (detections.who, and the recording's tags).

Clothing-based visual search can't do this: it matched anyone in a similar shirt. Faces separate cleanly: tested
on this footage, the same person scored 0.68 across two moments, everyone else 0.19 or less (SFace's usual
same-person line is 0.363).
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("theeye.faces")

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"
MIN_FACE_PX = 40          # smaller faces carry too little detail to recognise
MATCH = 0.42              # tag a face with a person at this similarity to their confirmed faces
SUGGEST = 0.30            # show as "is this ...?" at this similarity


class FaceEngine:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._det = self._rec = None
        self._lock = threading.Lock()
        self.ok: Optional[bool] = None

    def _load(self) -> bool:
        if self.ok is None:
            d, r = self.models_dir / DETECTOR, self.models_dir / RECOGNIZER
            if not (d.is_file() and r.is_file()):
                log.warning("face models not found in %s: face recognition off", self.models_dir)
                self.ok = False
            else:
                self._det = cv2.FaceDetectorYN.create(str(d), "", (320, 320), 0.8, 0.3, 5000)
                self._rec = cv2.FaceRecognizerSF.create(str(r), "")
                self.ok = True
                log.info("face recognition ready")
        return bool(self.ok)

    def best_face(self, img: np.ndarray, bbox: list[float]) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
        """The clearest face inside (a generous margin around) a person's box: (fingerprint, aligned 112x112 face,
        face size in px), or None if there's no usable face."""
        if img is None or len(bbox) != 4:
            return None
        h, w = img.shape[:2]
        x1, y1 = int(max(0.0, bbox[0] - 0.05) * w), int(max(0.0, bbox[1] - 0.05) * h)
        x2, y2 = int(min(1.0, bbox[2] + 0.05) * w), int(min(1.0, bbox[3] + 0.02) * h)
        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < MIN_FACE_PX or crop.shape[1] < MIN_FACE_PX:
            return None
        with self._lock:
            if not self._load():
                return None
            self._det.setInputSize((crop.shape[1], crop.shape[0]))
            _, found = self._det.detect(crop)
            if found is None:
                return None
            best = max(found, key=lambda f: float(f[-1]) * min(f[2], f[3]))
            size = int(min(best[2], best[3]))
            if size < MIN_FACE_PX:
                return None
            aligned = self._rec.alignCrop(crop, best)
            feat = self._rec.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) or 1.0), aligned, size


def similarity(a: np.ndarray, refs: np.ndarray) -> float:
    """Best cosine similarity of fingerprint a to any of the (normalised) reference fingerprints."""
    return float((refs @ a).max()) if len(refs) else 0.0
