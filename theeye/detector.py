"""Object detection (YOLO), lightweight per-camera tracking, plate reading and embeddings.

Frames arrive from cameras at analyze_fps; only the newest frame per camera is kept
(latest-wins), and inference is batched across cameras on the GPU.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .config import DetectorConfig
from .motion import Zone

log = logging.getLogger("theeye.detector")

VEHICLES = {"car", "truck", "bus", "motorcycle"}
TRACK_TTL = 3.0          # seconds unseen before a track is finalized
PLATE_RETRY_MAX_S = 30.0          # slowest retry for a vehicle whose plate hasn't read confidently
IOU_MATCH = 0.3


@dataclass
class Track:
    key: str
    camera_id: str
    label: str
    bbox: list[float]                 # normalized x1,y1,x2,y2 (latest)
    conf: float
    first_ts: float
    last_ts: float
    zones: set[int] = field(default_factory=set)
    best_conf: float = 0.0
    best_bbox: list[float] = field(default_factory=list)
    best_crop: Optional[np.ndarray] = None
    best_frame: Optional[np.ndarray] = None
    best_ts: float = 0.0                  # when best_frame was seen (for full-res re-decode from the RAM loop)
    snapshot_dirty: bool = False
    plate: Optional[str] = None
    plate_conf: float = 0.0
    last_plate_try: float = 0.0
    plate_tries: int = 0
    db_id: Optional[int] = None
    hits: int = 0

    def public(self) -> dict:
        return {"key": self.key, "label": self.label, "bbox": [round(v, 4) for v in self.bbox],
                "conf": round(self.conf, 3), "zones": sorted(self.zones), "plate": self.plate,
                "first_ts": self.first_ts, "last_ts": self.last_ts, "db_id": self.db_id}


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class Detector:
    def __init__(self, cfg: DetectorConfig, snap_dir: Path,
                 zones_for: Callable[[str], list[Zone]],
                 main_frame: Callable[[str], Optional[np.ndarray]],
                 embedder=None):
        self.cfg = cfg
        self.snap_dir = snap_dir
        self.zones_for = zones_for
        self.main_frame = main_frame
        self.embedder = embedder
        self._pending: dict[str, tuple[np.ndarray, float]] = {}
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.tracks: dict[str, dict[str, Track]] = {}
        self._tlock = threading.Lock()
        self._seq = 0
        self._model = None
        self._names: dict[int, str] = {}
        self._class_ids: Optional[list[int]] = None
        self._alpr = None
        # callbacks wired by the Hub
        self.on_track_new: Callable[[Track], None] = lambda t: None
        self.on_track_update: Callable[[Track], None] = lambda t: None
        self.on_track_end: Callable[[Track], None] = lambda t: None
        self.on_frame_tracks: Callable[[str, float, list[Track]], None] = lambda c, ts, t: None
        self.stats = {"infer_ms": 0.0, "batches": 0}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        threading.Thread(target=self._run, name="detector", daemon=True).start()
        threading.Thread(target=self._reaper, name="track-reaper", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def submit(self, cam: str, img: np.ndarray, ts: float) -> None:
        with self._cv:
            self._pending[cam] = (img, ts)        # latest-wins
            self._cv.notify()

    def active_tracks(self, cam: str, within: float = 1.5) -> list[Track]:
        now = time.time()
        with self._tlock:
            tracks = list(self.tracks.get(cam, {}).values())
        return [t for t in tracks if now - t.last_ts <= within]

    # ----------------------------------------------------------------- model
    def _load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.cfg.model)
        self._names = self._model.names
        wanted = set(self.cfg.classes)
        self._class_ids = [i for i, n in self._names.items() if n in wanted] or None
        log.info("YOLO %s loaded (%d classes tracked)", self.cfg.model, len(self._class_ids or []))
        if self.cfg.plates:
            try:
                from fast_alpr import ALPR
                import onnxruntime as ort
                cpu = ["CPUExecutionProvider"]          # plate models are tiny; CPU avoids CUDA-version clashes

                def sess_opts():
                    # ORT defaults to one spinning thread per core; between our sporadic calls those threads
                    # busy-wait and burned several cores. Two non-spinning threads are plenty for these models.
                    so = ort.SessionOptions()
                    so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
                    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
                    return so
                self._alpr = ALPR(detector_model=self.cfg.plate_detector, ocr_model=self.cfg.plate_ocr,
                                  detector_providers=cpu, ocr_providers=cpu, ocr_device="cpu",
                                  detector_sess_options=sess_opts(), ocr_sess_options=sess_opts())
                log.info("ALPR loaded")
            except Exception:  # noqa: BLE001
                log.exception("ALPR failed to load; plates disabled")

    def _run(self) -> None:
        try:
            self._load()
        except Exception:  # noqa: BLE001
            log.exception("detector failed to load; detection disabled")
            return
        while not self._stop.is_set():
            with self._cv:
                self._cv.wait_for(lambda: self._pending or self._stop.is_set(), timeout=1.0)
                batch = list(self._pending.items())
                self._pending.clear()
            if not batch:
                continue
            t0 = time.perf_counter()
            try:
                results = self._model.predict([b[1][0] for b in batch], conf=self.cfg.confidence,
                                              classes=self._class_ids, device=self.cfg.device,
                                              verbose=False)
            except Exception:  # noqa: BLE001
                log.exception("inference failed")
                time.sleep(1)
                continue
            self.stats["infer_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self.stats["batches"] += 1
            for (cam, (img, ts)), res in zip(batch, results):
                try:
                    with self._tlock:                       # only track-table mutation under the lock
                        new, upd, seen, plate_jobs = self._handle(cam, img, ts, res)
                    for t in plate_jobs:                    # slow: decodes main-stream GOP
                        self._read_plate(cam, t)
                    for t in new:
                        self.on_track_new(t)
                    for t in upd:
                        self.on_track_update(t)
                    self.on_frame_tracks(cam, ts, seen)
                except Exception:  # noqa: BLE001
                    log.exception("[%s] post-processing failed", cam)

    # -------------------------------------------------------------- tracking
    def _handle(self, cam: str, img: np.ndarray, ts: float, res):
        """Match detections to tracks. Pure state update; caller runs callbacks outside the lock."""
        h, w = img.shape[:2]
        zones = [z for z in self.zones_for(cam) if z.enabled]
        excl = [z for z in zones if z.kind == "exclude"]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = (b.xyxy[0].tolist())
            nb = [x1 / w, y1 / h, x2 / w, y2 / h]
            ax, ay = (nb[0] + nb[2]) / 2, nb[3]           # anchor = bottom-centre ("feet")
            if any(z.contains(ax, ay) for z in excl):
                continue
            label = self._names[int(b.cls[0])]
            zids = {z.id for z in zones if z.kind != "exclude" and z.contains(ax, ay)}
            dets.append((label, float(b.conf[0]), nb, zids))

        cam_tracks = self.tracks.setdefault(cam, {})
        unmatched = set(cam_tracks)
        seen: list[Track] = []
        new: list[Track] = []
        upd: list[Track] = []
        plate_jobs: list[Track] = []
        for label, conf, nb, zids in sorted(dets, key=lambda d: -d[1]):
            best, best_iou = None, IOU_MATCH
            for k in unmatched:
                t = cam_tracks[k]
                if t.label != label:
                    continue
                v = _iou(t.bbox, nb)
                if v > best_iou:
                    best, best_iou = t, v
            if best is None:
                self._seq += 1
                best = Track(f"{cam}:{int(ts)}:{self._seq}", cam, label, nb, conf, ts, ts)
                cam_tracks[best.key] = best
                is_new = True
            else:
                unmatched.discard(best.key)
                is_new = False
            best.bbox, best.conf, best.last_ts, best.hits = nb, conf, ts, best.hits + 1
            best.zones |= zids
            if conf > best.best_conf * 1.05:
                best.best_conf, best.best_bbox = conf, nb
                best.best_crop = _crop(img, nb, pad=0.15)
                best.best_frame = img.copy()
                best.best_ts = ts
                best.snapshot_dirty = True
            # Each try decodes a full-res GOP, so back off (1,2,4..30 s) on plates that won't read, e.g. parked cars.
            if (self._alpr and label in VEHICLES and best.plate_conf < 0.9
                    and ts - best.last_plate_try > min(PLATE_RETRY_MAX_S, 2.0 ** best.plate_tries)):
                best.last_plate_try = ts
                best.plate_tries += 1
                plate_jobs.append(best)
            (new if is_new else upd).append(best)
            seen.append(best)
        return new, upd, seen, plate_jobs

    def _read_plate(self, cam: str, t: Track) -> None:
        frame = self.main_frame(cam)
        if frame is None:
            return
        crop = _crop(frame, t.bbox, pad=0.1)
        if crop is None or crop.shape[0] < 40 or crop.shape[1] < 40:
            return
        try:
            for r in self._alpr.predict(crop):
                if not r.ocr or not r.ocr.text:
                    continue
                c = r.ocr.confidence
                c = float(np.mean(c)) if isinstance(c, (list, tuple, np.ndarray)) else float(c)
                c *= float(r.detection.confidence)
                text = "".join(ch for ch in r.ocr.text.upper() if ch.isalnum())
                if len(text) >= 3 and c > t.plate_conf:
                    t.plate, t.plate_conf = text, c
        except Exception:  # noqa: BLE001
            log.debug("plate read failed", exc_info=True)

    def _reaper(self) -> None:
        while not self._stop.wait(1.0):
            now = time.time()
            ended: list[Track] = []
            with self._tlock:
                for tr in self.tracks.values():
                    for k, t in list(tr.items()):
                        if now - t.last_ts > TRACK_TTL:
                            ended.append(tr.pop(k))
            for t in ended:            # outside the lock: snapshot/embedding work is slow
                try:
                    self.on_track_end(t)
                except Exception:  # noqa: BLE001
                    log.exception("track end handler failed")

    # -------------------------------------------------------------- outputs
    def save_snapshot(self, t: Track) -> Optional[str]:
        """Write best frame (with box) to disk; returns path relative to data dir."""
        if t.best_frame is None:
            return None
        img = t.best_frame.copy()
        h, w = img.shape[:2]
        x1, y1, x2, y2 = t.best_bbox
        cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 200, 255), 2)
        txt = t.label + (f" {t.plate}" if t.plate else "")
        cv2.putText(img, txt, (int(x1 * w), max(14, int(y1 * h) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        day = time.strftime("%Y-%m-%d", time.localtime(t.first_ts))
        rel = Path("snaps") / t.camera_id / day / f"det_{t.key.replace(':', '_')}.jpg"
        out = self.snap_dir.parent / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        t.snapshot_dirty = False
        return str(rel)

    def embed(self, t: Track) -> Optional[bytes]:
        if not (self.embedder and self.cfg.embeddings) or t.best_crop is None:
            return None
        try:
            v = self.embedder.image(t.best_crop)
            return v.tobytes() if v is not None else None
        except Exception:  # noqa: BLE001
            log.exception("embedding failed")
            return None


def _crop(img: np.ndarray, nb: list[float], pad: float) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    bw, bh = nb[2] - nb[0], nb[3] - nb[1]
    x1 = max(0, int((nb[0] - bw * pad) * w))
    y1 = max(0, int((nb[1] - bh * pad) * h))
    x2 = min(w, int((nb[2] + bw * pad) * w))
    y2 = min(h, int((nb[3] + bh * pad) * h))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img[y1:y2, x1:x2].copy()
