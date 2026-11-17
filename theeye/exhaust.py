"""Loud exhaust: a car revving or roaring past, told apart from ordinary road noise and the gardener's blower.

Each finished recording from a camera with a microphone is listened to after it closes:
  1. YAMNet (Google's AudioSet sound classifier, 521 sounds, a small TFLite model on the CPU) scores every
     ~1 s of audio. The clip is first levelled (these mics record very quietly).
  2. At each loud moment (a local loudness peak at -32 dBFS or more), it adds up the exhaust-like scores
     (revving, race car, motorcycle, roar) over +-3 s, and needs a car-type sound there too. Any blower / lawn
     mower / power tool sound within +-15 s rules the moment out.
  3. The frames around that moment must show a vehicle driving past (YOLO on the recording itself: fast cars
     cross the frame in under a second and often never become a detection).
A moment that passes tags the recording "loud exhaust" (searchable) and adds a "loud exhaust" sighting of the car,
matched to a vehicle where it can be. A vehicle loud on `vehicles.exhaust_visits` separate visits is flagged.

Tested on 3 days of street footage (4,288 recordings): the loud exhausts were found, blowers and road noise not.
"""
from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
import numpy as np

log = logging.getLogger("theeye.exhaust")

TAG = "loud exhaust"
MODEL_DIR = "yamnet"                     # <install>/models/yamnet/{yamnet.tflite, yamnet_class_map.csv}
RATE, WIN, HOP = 16000, 15600, 7800      # YAMNet input: 0.975 s at 16 kHz; half-window hop
LOUD_DB = -32.0                          # a moment must peak at least this loud (raw clip level)
EXH = ["Accelerating, revving, vroom", "Race car, auto racing", "Motorcycle", "Roar", "Roaring cats (lions, tigers)"]
VEH = ["Vehicle", "Car", "Motor vehicle (road)"]
BLOW = ["Lawn mower", "Power tool", "Tools", "Jackhammer", "Chainsaw", "Drill"]
MIN_VEH = 0.4                            # car-type sound near the moment
MAX_BLOW = 0.15                          # blower / tool sound anywhere within +-15 s
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class ExhaustWatch:
    def __init__(self, hub, models_dir: Path):
        self.hub, self.db = hub, hub.db
        self.dir = models_dir / MODEL_DIR
        self._it = self._yolo = None
        self._idx: dict[str, list[int]] = {}
        self.ok: Optional[bool] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        threading.Thread(target=self._loop, name="exhaust", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _load(self) -> bool:
        if self.ok is None:
            model, names = self.dir / "yamnet.tflite", self.dir / "yamnet_class_map.csv"
            try:
                from ai_edge_litert.interpreter import Interpreter
                if not (model.is_file() and names.is_file()):
                    raise FileNotFoundError(f"{model} / {names}")
                self._it = Interpreter(model_path=str(model), num_threads=1)
                self._it.allocate_tensors()
                self._in = self._it.get_input_details()[0]["index"]
                self._out = self._it.get_output_details()[0]["index"]
                with open(names) as f:
                    cls = [r["display_name"] for r in csv.DictReader(f)]
                self._idx = {k: [cls.index(n) for n in group] for k, group in (("exh", EXH), ("veh", VEH), ("blow", BLOW))}
                self.ok = True
                log.info("loud exhaust detection ready")
            except Exception as e:  # noqa: BLE001
                log.warning("loud exhaust detection off: %s", e)
                self.ok = False
        return bool(self.ok)

    def _cameras(self) -> list[str]:
        want = set(self.hub.cfg.exhaust.cameras)
        return [cid for cid, cam in self.hub.cams.items() if cam.has_audio and (not want or cid in want)]

    def _loop(self) -> None:
        if self._stop.wait(30) or not self.hub.cfg.exhaust.enabled or not self._load():
            return
        while not self._stop.is_set():
            busy = False
            try:
                busy = self._pass()
            except Exception:  # noqa: BLE001
                log.exception("loud exhaust pass failed")
            self._stop.wait(2 if busy else 60)

    def _pass(self) -> bool:
        """Look at a batch of finished, not yet scanned recordings, newest first (so new ones are quick and older
        ones fill in behind). Returns whether there may be more."""
        cams = self._cameras()
        if not cams:
            return False
        active = {w["path"] for w in self.hub.rec.active().values()}
        rows = self.db.all(
            "SELECT clip_path, camera_id, MIN(start_ts - clip_offset) AS clip_start FROM events"
            f" WHERE clip_path IS NOT NULL AND end_ts IS NOT NULL AND camera_id IN ({','.join('?' * len(cams))})"
            " AND clip_path NOT IN (SELECT clip_path FROM sound_scans)"
            " GROUP BY clip_path ORDER BY MAX(start_ts) DESC LIMIT 20", cams)
        rows = [r for r in rows if r["clip_path"] not in active]
        for r in rows:
            if self._stop.is_set():
                break
            found = 0
            try:
                found = self.scan(r["clip_path"], r["camera_id"], r["clip_start"])
            except Exception:  # noqa: BLE001
                log.exception("loud exhaust: %s failed", r["clip_path"])
            self.db.execute("INSERT OR REPLACE INTO sound_scans(clip_path, ts, found) VALUES(?,?,?)",
                            (r["clip_path"], time.time(), found))
        return len(rows) == 20

    # ----------------------------------------------------------------- sound
    def _audio(self, path: Path) -> Optional[np.ndarray]:
        with av.open(str(path)) as c:
            if not c.streams.audio:
                return None
            rs = av.AudioResampler(format="flt", layout="mono", rate=RATE)
            buf = []
            for pkt in c.demux(c.streams.audio[0]):
                try:
                    frames = pkt.decode()
                except av.error.FFmpegError:          # a damaged packet: skip it, keep the rest
                    continue
                buf += [g.to_ndarray().reshape(-1) for f in frames for g in rs.resample(f)]
        return np.concatenate(buf) if buf else None

    def moments(self, x: np.ndarray) -> list[tuple[float, float, float]]:
        """Loud exhaust-like moments in mono 16 kHz audio: [(seconds, score, dBFS)], best first, 10 s apart."""
        if len(x) < WIN:
            return []
        gain = 0.1 / (float(np.sqrt(np.mean(x ** 2))) + 1e-9)       # level the clip (to about -20 dBFS)
        T, D, S = [], [], {k: [] for k in self._idx}
        for s in range(0, len(x) - WIN + 1, HOP):
            w = x[s:s + WIN]
            self._it.set_tensor(self._in, (w * gain).astype(np.float32))
            self._it.invoke()
            sc = self._it.get_tensor(self._out)[0]
            T.append(s / RATE)
            D.append(20 * np.log10(float(np.sqrt(np.mean(w ** 2))) + 1e-9))
            for k, ix in self._idx.items():
                S[k].append(float(sc[ix].max()))
        T, D = np.array(T), np.array(D)
        exh, veh, blow = (np.array(S[k]) for k in ("exh", "veh", "blow"))
        min_score = self.hub.cfg.exhaust.min_score
        out = []
        for i in range(len(T)):
            if D[i] < LOUD_DB or (i and D[i - 1] > D[i]) or (i + 1 < len(T) and D[i + 1] > D[i]):
                continue                                  # only loud local peaks
            near, wide = np.abs(T - T[i]) <= 3, np.abs(T - T[i]) <= 15
            score = float(exh[near].sum())
            if score >= min_score and veh[near].max() >= MIN_VEH and blow[wide].max() < MAX_BLOW:
                out.append((float(T[i]), score, float(D[i])))
        out.sort(key=lambda m: -m[1])
        kept: list[tuple[float, float, float]] = []
        for m in out:
            if all(abs(m[0] - k[0]) > 10 for k in kept):
                kept.append(m)
        return kept

    # ---------------------------------------------------------------- vision
    def _frames(self, path: Path, t: float) -> list[np.ndarray]:
        want = [t + d for d in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)]
        out = []
        with av.open(str(path)) as c:
            v = c.streams.video[0]
            c.seek(int(max(0.0, t - 3) / v.time_base), stream=v)
            for f in c.decode(v):
                if f.time is None:
                    continue
                if want and f.time >= want[0]:
                    out.append(f.to_ndarray(format="bgr24", width=1280, height=720))
                    want.pop(0)
                if not want:
                    break
        return out

    def passing(self, path: Path, t: float):
        """The vehicle driving past at clip time t: (frame, normalised box, label, confidence), or None. A box
        counts as driving past when nothing overlaps it much in the frames half a second either side."""
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self.hub.cfg.detector.model)
        frames = self._frames(path, t)
        boxes = []
        for im in frames:
            r = self._yolo.predict(im, conf=0.35, verbose=False, device="cpu", classes=list(COCO_VEHICLES))[0]
            boxes.append([(b, COCO_VEHICLES[int(c)], float(p)) for b, c, p in
                          zip(r.boxes.xyxyn.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())])
        best = None
        for i, bs in enumerate(boxes):
            others = [o[0] for j in (i - 1, i + 1) if 0 <= j < len(boxes) for o in boxes[j]]
            for b, label, p in bs:
                if b[2] - b[0] > 0.03 and not any(_iou(b, o) > 0.5 for o in others):
                    size = (b[2] - b[0]) * (b[3] - b[1])
                    if best is None or size > best[0]:
                        best = (size, frames[i], b, label, p)
        return best[1:] if best else None

    # ------------------------------------------------------------------ scan
    def scan(self, rel: str, cam: str, clip_start: float) -> int:
        """Find loud exhausts in one recording and record them. Returns how many."""
        path = self.hub.data / rel
        if not path.is_file() or not self._load():
            return 0
        x = self._audio(path)
        if x is None:
            return 0
        n = 0
        for t, score, db in self.moments(x):
            car = self.passing(path, t)
            if car is None:
                continue
            self._record(rel, cam, clip_start + t, t, score, db, car)
            n += 1
        return n

    def _record(self, rel: str, cam: str, ts: float, t: float, score: float, db: float, car) -> None:
        frame, box, label, conf = car
        ev = self.db.one("SELECT id FROM events WHERE clip_path=? AND start_ts <= ? + 2"
                         " AND COALESCE(end_ts, start_ts) >= ? - 2 ORDER BY start_ts LIMIT 1", (rel, ts, ts)) \
            or self.db.one("SELECT id FROM events WHERE clip_path=? ORDER BY ABS(start_ts - ?) LIMIT 1", (rel, ts))
        eid = ev["id"] if ev else None
        vlabels = self.hub.cfg.vehicles.labels
        det = self.db.one(
            f"SELECT id, vehicle_id, subtype FROM detections WHERE camera_id=? AND label IN ({','.join('?' * len(vlabels))})"
            " AND moving=1 AND first_ts <= ? + 2 AND last_ts >= ? - 2 ORDER BY ABS((first_ts + last_ts) / 2 - ?) LIMIT 1",
            [cam, *vlabels, ts, ts, ts])
        if det:                                   # the detector did see this car: tag that sighting
            det_id, vid = det["id"], det["vehicle_id"]
            if not det["subtype"]:
                self.db.execute("UPDATE detections SET subtype=?, subtype_conf=? WHERE id=?", (TAG, round(score, 3), det_id))
        else:                                     # too fast for the detector: add the sighting from the recording
            det_id, vid = self._add_sighting(cam, rel, ts, t, score, frame, box, label, conf, eid)
        if eid:
            self._tag_event(eid)
        self.db.execute("INSERT INTO loud_exhausts(camera_id, ts, score, db, event_id, detection_id, vehicle_id, clip_path, t)"
                        " VALUES(?,?,?,?,?,?,?,?,?)", (cam, ts, round(score, 3), round(db, 1), eid, det_id, vid, rel, t))
        flagged = self.hub.vehicles.loud_exhaust(vid) if vid else False
        log.info("[%s] loud exhaust at %s (score %.2f, %.0f dB, %s%s)", cam, time.strftime("%H:%M:%S", time.localtime(ts)),
                 score, db, f"vehicle {vid}" if vid else "vehicle not matched", ", flagged" if flagged else "")

    def _add_sighting(self, cam, rel, ts, t, score, frame, box, label, conf, eid):
        h, w = frame.shape[:2]
        crop = frame[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
        emb = None
        if self.hub.embedder is not None and crop.size:
            try:
                v = self.hub.embedder.image(crop)
                emb = v.tobytes() if v is not None else None
            except Exception:  # noqa: BLE001
                log.exception("loud exhaust: embedding failed")
        img = frame.copy()
        cv2.rectangle(img, (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h)), (0, 200, 255), 2)
        cv2.putText(img, TAG, (int(box[0] * w), max(14, int(box[1] * h) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 1, cv2.LINE_AA)
        snap = Path("snaps") / cam / time.strftime("%Y-%m-%d", time.localtime(ts)) / f"exhaust_{int(ts * 10)}.jpg"
        (self.hub.data / snap).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.hub.data / snap), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        det_id = self.db.insert(
            "INSERT INTO detections(camera_id, track_key, label, confidence, first_ts, last_ts, bbox, zones, snapshot_path,"
            " embedding, event_id, moving, subtype, subtype_conf) VALUES(?,?,?,?,?,?,?,'[]',?,?,?,1,?,?)",
            (cam, f"exhaust:{rel}:{t:.1f}", label, round(conf, 3), ts - 1.5, ts + 1.5, json.dumps([round(v, 4) for v in box]),
             str(snap), emb, eid, TAG, round(score, 3)))
        if eid:
            self.db.execute("INSERT INTO event_links VALUES(?,?,?)", (eid, "detection", det_id))
        vid = None
        if self.hub.vehicles.eligible(label, box, True):
            try:
                vid = self.hub.vehicles.observe(det_id, cam, label, ts - 1.5, ts + 1.5, emb, None, 0.0, str(snap))
            except Exception:  # noqa: BLE001
                log.exception("loud exhaust: vehicle matching failed")
        return det_id, vid

    def _tag_event(self, eid: int) -> None:
        r = self.db.one("SELECT labels FROM events WHERE id=?", (eid,))
        if r is None:
            return
        labels = set(json.loads(r["labels"] or "[]"))
        if TAG not in labels:
            self.db.execute("UPDATE events SET labels=? WHERE id=?", (json.dumps(sorted(labels | {TAG})), eid))


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0
