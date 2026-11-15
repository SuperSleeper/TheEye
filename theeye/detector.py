"""Object detection (YOLO), lightweight per-camera tracking, plate reading and embeddings.

Frames arrive from cameras at analyze_fps; only the newest frame per camera is kept
(latest-wins), and inference is batched across cameras on the GPU.
"""
from __future__ import annotations

import json
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
# Things that legitimately sit still (parked vehicles, dropped bags). Shown and logged even when not moving,
# but they still only trigger rules once seen moving. Anything else (people, animals) that is never seen
# moving is treated as a misdetection and hidden entirely. COCO has no "package" class; bags are closest.
STILL_OK = VEHICLES | {"bicycle", "suitcase", "backpack", "handbag"}
TRACK_TTL = 3.0          # seconds unseen before a track is finalized
PLATE_RETRY_MAX_S = 30.0          # slowest retry for a vehicle whose plate hasn't read confidently
IOU_MATCH = 0.3
TRAVEL_IOU = 0.5         # box overlap with where the track began; below this it has actually driven somewhere
PARK_IOU = 0.8           # box overlap with where it last moved; above this counts as "hasn't moved"
PARK_MEMORY_S = 600.0    # remember parked spots this long, so a briefly occluded parked car stays parked


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
    still_bbox: list[float] = field(default_factory=list)   # where it was when it last moved
    still_since: float = 0.0
    parked: bool = False             # hasn't moved for detector.stationary_seconds: rules ignore it
    seen_moving: bool = False        # motion seen inside its box at some point (sticky for the track)
    start_bbox: list[float] = field(default_factory=list)   # where the track began
    travelled: bool = False          # the box itself left its starting spot (a parked car with a shadow or a
                                     # passer-by over it is "seen moving" but never travels)
    confirmed: bool = True           # counts for rules: seen moving, or motion confirmation is off
    announced: bool = False          # reported to the hub (Live, DB, search); withheld until plausible
    visible: bool = False            # currently shown/considered at all (see Detector._handle)
    db_id: Optional[int] = None
    hits: int = 0

    def public(self) -> dict:
        return {"key": self.key, "label": self.label, "bbox": [round(v, 4) for v in self.bbox],
                "conf": round(self.conf, 3), "zones": sorted(self.zones), "plate": self.plate,
                "first_ts": self.first_ts, "last_ts": self.last_ts, "db_id": self.db_id, "parked": self.parked,
                "confirmed": self.confirmed}


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
                 embedder=None,
                 motion_in_box: Optional[Callable[[str, list[float]], float]] = None):
        self.cfg = cfg
        self.snap_dir = snap_dir
        self.zones_for = zones_for
        self.main_frame = main_frame
        self.embedder = embedder
        self.motion_in_box = motion_in_box
        self._pending: dict[str, tuple[np.ndarray, float]] = {}
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.tracks: dict[str, dict[str, Track]] = {}
        self._parked_spots: dict[str, list[tuple[str, list[float], float]]] = {}   # cam -> (label, bbox, last seen)
        # Persisted so a restart doesn't make every parked car look newly arrived for stationary_seconds.
        self._spots_file = snap_dir.parent / "parked_spots.json"
        self._load_spots()
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
        self._thread: Optional[threading.Thread] = None
        self._busy_since: Optional[float] = None      # monotonic time the current batch started, None when idle

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="detector", daemon=True)
        self._thread.start()
        threading.Thread(target=self._reaper, name="track-reaper", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def stalled(self, limit_s: float) -> Optional[str]:
        """Why detection has stopped working, or None if it's fine. A GPU fault (e.g. NVIDIA Xid from another
        program) can leave a CUDA call hanging forever without raising, which silently stops all detection."""
        if self._thread is None or self._stop.is_set():
            return None
        if not self._thread.is_alive():
            return "thread has exited"
        busy = self._busy_since
        if busy is not None and time.monotonic() - busy > limit_s:
            return f"stuck on one batch for {time.monotonic() - busy:.0f} s"
        return None

    def submit(self, cam: str, img: np.ndarray, ts: float) -> None:
        with self._cv:
            self._pending[cam] = (img, ts)        # latest-wins
            self._cv.notify()

    def active_tracks(self, cam: str, within: float = 1.5) -> list[Track]:
        now = time.time()
        with self._tlock:
            tracks = list(self.tracks.get(cam, {}).values())
        return [t for t in tracks if t.visible and now - t.last_ts <= within]

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
            self._busy_since = time.monotonic()
            try:
                self._process(batch)
            finally:
                self._busy_since = None

    def _process(self, batch: list) -> None:
        t0 = time.perf_counter()
        try:
            results = self._model.predict([b[1][0] for b in batch], conf=self.cfg.confidence,
                                          classes=self._class_ids, device=self.cfg.device,
                                          verbose=False)
        except Exception:  # noqa: BLE001
            log.exception("inference failed")
            time.sleep(1)
            return
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
        # Zones marked "only detect objects inside": if any exist, objects outside all of them are ignored.
        ai_zones = [z for z in zones if z.kind != "exclude" and (z.ai or z.kind == "detect")]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = (b.xyxy[0].tolist())
            nb = [x1 / w, y1 / h, x2 / w, y2 / h]
            ax, ay = (nb[0] + nb[2]) / 2, nb[3]           # anchor = bottom-centre ("feet")
            if any(z.contains(ax, ay) for z in excl):
                continue
            if ai_zones and not any(z.contains(ax, ay) for z in ai_zones):
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
                best.still_bbox, best.still_since, best.start_bbox = nb, ts, nb
                if self._was_parked_here(cam, label, nb, ts):   # reappeared in a known parked spot
                    best.still_since = ts - self.cfg.stationary_seconds
                cam_tracks[best.key] = best
            else:
                unmatched.discard(best.key)
            best.bbox, best.conf, best.last_ts, best.hits = nb, conf, ts, best.hits + 1
            if not best.travelled and _iou(best.start_bbox, nb) < TRAVEL_IOU:
                best.travelled = True
            if _iou(best.still_bbox, nb) < PARK_IOU:          # it moved: restart the stillness clock
                best.still_bbox, best.still_since = nb, ts
            best.parked = self.cfg.stationary_seconds > 0 and ts - best.still_since >= self.cfg.stationary_seconds
            if not best.seen_moving and self.motion_in_box is not None:
                best.seen_moving = self.motion_in_box(cam, nb) >= self.cfg.motion_min_fraction
            best.confirmed = best.seen_moving or not self.cfg.require_motion or self.motion_in_box is None
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
            # Shown only while plausible: things that are normally still (vehicles, bags) always; anything
            # else (people, animals) only once seen moving and while not stationary - a "person" that never
            # moves, or stops moving for stationary_seconds, is almost always a misread static object.
            best.visible = (not self.cfg.require_motion or best.label in STILL_OK
                            or (best.seen_moving and not best.parked))
            if best.visible:
                (upd if best.announced else new).append(best)
                best.announced = True
                seen.append(best)
        return new, upd, seen, plate_jobs

    def _was_parked_here(self, cam: str, label: str, bbox: list[float], ts: float) -> bool:
        if self.cfg.stationary_seconds <= 0:
            return False
        spots = [s for s in self._parked_spots.get(cam, []) if ts - s[2] < PARK_MEMORY_S]
        self._parked_spots[cam] = spots
        return any(lbl == label and _iou(b, bbox) >= PARK_IOU for lbl, b, _ in spots)

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

    def _load_spots(self) -> None:
        try:
            raw = json.loads(self._spots_file.read_text())
            now = time.time()
            self._parked_spots = {cam: [(l, b, ts) for l, b, ts in spots if now - ts < PARK_MEMORY_S]
                                  for cam, spots in raw.items()}
        except (OSError, ValueError, TypeError):
            self._parked_spots = {}

    def _save_spots(self) -> None:
        """Remembered spots plus objects parked right now (they haven't 'ended', but a restart would lose them)."""
        with self._tlock:
            out = {cam: list(spots) for cam, spots in self._parked_spots.items()}
            for cam, tr in self.tracks.items():
                out.setdefault(cam, []).extend((t.label, t.bbox, t.last_ts) for t in tr.values() if t.parked)
        try:
            tmp = self._spots_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(out))
            tmp.replace(self._spots_file)
        except OSError:
            log.debug("could not save parked spots", exc_info=True)

    def _reaper(self) -> None:
        last_save = 0.0
        while not self._stop.wait(1.0):
            if time.time() - last_save > 10:
                self._save_spots()
                last_save = time.time()
            now = time.time()
            ended: list[Track] = []
            with self._tlock:
                for tr in self.tracks.values():
                    for k, t in list(tr.items()):
                        if now - t.last_ts > TRACK_TTL:
                            ended.append(tr.pop(k))
                for t in ended:            # remember where parked objects were, in case they reappear
                    if t.parked:
                        self._parked_spots.setdefault(t.camera_id, []).append((t.label, t.bbox, t.last_ts))
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
