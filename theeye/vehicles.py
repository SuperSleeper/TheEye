"""Repeat-vehicle tracking.

Each travelling vehicle sighting (a finished detection track whose box drove somewhere; parked cars never count) is matched to a known vehicle by licence
plate when one was read confidently, otherwise by appearance (CLIP embedding of the best crop).
Sightings of the same vehicle more than `visit_gap_minutes` apart count as separate visits; a vehicle
reaching `repeat_visits` is flagged. Vehicles marked "known" (your own cars, neighbours) never flag.

Measured on real footage (Wyze, 1440p): same parked car pairs median cosine 0.93, different cars 0.44;
at 0.88, 82% of same-car pairs match and 0.02% of different-car pairs do.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .db import DB, jload

log = logging.getLogger("theeye.vehicles")

GROUP = {"car": "car", "truck": "car", "bus": "car", "motorcycle": "motorcycle", "bicycle": "bicycle"}
MIN_AREA = 0.01           # smaller boxes are mostly background (same street/tree), so different cars look alike
MAX_EXEMPLARS = 8         # stored sightings per vehicle; a new one must resemble them ON AVERAGE (no chaining)


@dataclass
class _V:
    id: int
    group: str
    plate: Optional[str]
    ex: Optional[np.ndarray]           # (k, d) exemplar embeddings
    n: int
    visits: int
    last_ts: float
    status: str
    cameras: set[str] = field(default_factory=set)


class VehicleRegistry:
    def __init__(self, db: DB, cfg):
        self.db, self.cfg = db, cfg
        self._lock = threading.Lock()
        self._v: dict[int, _V] = {}
        for r in db.all("SELECT * FROM vehicles"):
            ex = (np.frombuffer(r["embedding"], dtype=np.float32).reshape(r["n_ex"], -1).copy()
                  if r["embedding"] and r["n_ex"] else None)
            self._v[r["id"]] = _V(r["id"], r["grp"], r["plate"], ex,
                                  r["sightings"], r["visits"], r["last_ts"], r["status"], set(jload(r["cameras"], [])))

    # ------------------------------------------------------------------ match
    def eligible(self, label: str, bbox: list[float], moving: bool) -> bool:
        if not self.cfg.enabled or label not in self.cfg.labels or not moving or len(bbox) != 4:
            return False
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= MIN_AREA

    def observe(self, det_id: int, camera_id: str, label: str, ts_first: float, ts_last: float,
                emb: Optional[bytes], plate: Optional[str], plate_conf: float,
                snapshot: Optional[str]) -> Optional[int]:
        """Attach one finished sighting to a vehicle (creating it if new). Returns the vehicle id."""
        group = GROUP.get(label, label)
        vec = np.frombuffer(emb, dtype=np.float32) if emb else None
        plate = plate if (plate and plate_conf >= 0.6 and len(plate) >= 5) else None
        if vec is None and plate is None:
            return None
        with self._lock:
            v = self._match(group, vec, plate)
            if v is None:
                vid = self.db.insert(
                    "INSERT INTO vehicles(grp,label,plate,embedding,n_ex,sightings,visits,first_ts,last_ts,cameras,"
                    "snapshot_path,status) VALUES(?,?,?,?,?,1,1,?,?,?,?,'new')",
                    (group, label, plate, vec.tobytes() if vec is not None else None, int(vec is not None),
                     ts_first, ts_last,
                     json.dumps([camera_id]), snapshot))
                v = self._v[vid] = _V(vid, group, plate, vec[None, :].copy() if vec is not None else None, 1, 1,
                                      ts_last, "new", {camera_id})
            else:
                new_visit = ts_first - v.last_ts > self.cfg.visit_gap_minutes * 60
                v.visits += int(new_visit)
                v.n += 1
                v.last_ts = max(v.last_ts, ts_last)
                v.cameras.add(camera_id)
                v.plate = v.plate or plate
                if vec is not None:
                    if v.ex is None:
                        v.ex = vec[None, :].copy()
                    elif len(v.ex) < MAX_EXEMPLARS:
                        v.ex = np.vstack([v.ex, vec])
                    else:                               # keep the first sightings, refresh the newest slot
                        v.ex[-1] = vec
                flag = v.status == "new" and v.visits >= self.cfg.repeat_visits
                if flag:
                    v.status = "flagged"
                    log.info("vehicle %d flagged: %d visits (%s)", v.id, v.visits, ", ".join(sorted(v.cameras)))
                self.db.execute(
                    "UPDATE vehicles SET sightings=?, visits=?, last_ts=?, cameras=?, plate=?, embedding=?, n_ex=?, status=?,"
                    " flagged_ts=CASE WHEN ? THEN ? ELSE flagged_ts END,"
                    " snapshot_path=COALESCE(?, snapshot_path) WHERE id=?",
                    (v.n, v.visits, v.last_ts, json.dumps(sorted(v.cameras)), v.plate,
                     v.ex.tobytes() if v.ex is not None else None, len(v.ex) if v.ex is not None else 0,
                     v.status, int(flag), ts_last,
                     snapshot if new_visit else None, v.id))
            self.db.execute("UPDATE detections SET vehicle_id=? WHERE id=?", (v.id, det_id))
            return v.id

    def _match(self, group: str, vec: Optional[np.ndarray], plate: Optional[str]) -> Optional[_V]:
        if plate:
            for v in self._v.values():
                if v.plate and _close_plates(v.plate, plate):
                    return v
        if vec is None:
            return None
        best, best_sim = None, self.cfg.match_similarity
        for v in self._v.values():
            if v.group != group or v.ex is None or v.ex.shape[1] != vec.shape[0]:
                continue
            s = float((v.ex @ vec).mean())              # average over exemplars: one look-alike isn't enough
            if s >= best_sim:
                best, best_sim = v, s
        return best

    # ------------------------------------------------------------------ admin
    def set_status(self, vid: int, status: str, note: Optional[str] = None) -> None:
        with self._lock:
            v = self._v.get(vid)
            if v is None:
                raise KeyError(vid)
            # Unflagging restarts the count (it flags again after `repeat_visits` more visits);
            # "Not known" keeps the history.
            if status == "new" and v.status == "flagged":
                v.visits = 0
            v.status = status
            self.db.execute("UPDATE vehicles SET status=?, visits=?, note=COALESCE(?, note) WHERE id=?",
                            (status, v.visits, note, vid))

    def set_plate(self, vid: int, plate: str, conf: float) -> None:
        """A plate read more surely later (theeye.plates), e.g. over the whole recording."""
        if conf < 0.6 or len(plate) < 5:
            return
        with self._lock:
            v = self._v.get(vid)
            if v is None or v.plate == plate:
                return
            v.plate = plate
            self.db.execute("UPDATE vehicles SET plate=? WHERE id=?", (plate, vid))

    def backfill(self) -> int:
        """Seed from past detections. Ones recorded with travel info use it; older ones didn't store it, so a
        box that sits where vehicles were also seen on PARKED_REPEATS other visits is taken as a parked car."""
        rows = self.db.all("SELECT id, camera_id, label, first_ts, last_ts, bbox, embedding, plate, plate_conf,"
                           " snapshot_path, moving FROM detections WHERE vehicle_id IS NULL AND last_ts IS NOT NULL"
                           " ORDER BY first_ts")
        rows = [r for r in rows if r["label"] in self.cfg.labels]
        parked = _parked_spots(rows, self.cfg.visit_gap_minutes * 60)
        n = 0
        for r in rows:
            moving = bool(r["moving"]) if r["moving"] is not None else r["id"] not in parked
            if self.eligible(r["label"], jload(r["bbox"], []), moving):
                if self.observe(r["id"], r["camera_id"], r["label"], r["first_ts"], r["last_ts"], r["embedding"],
                                r["plate"], r["plate_conf"] or 0.0, r["snapshot_path"]):
                    n += 1
        return n


PARKED_IOU = 0.8          # boxes overlapping this much are the same spot
PARKED_REPEATS = 2        # seen in the same spot on this many other visits: a parking spot, not a passing car


def _parked_spots(rows: list, gap_s: float) -> set[int]:
    """Ids of detections without travel info whose box recurs at the same spot on other visits."""
    by_cam: dict[str, list] = {}
    for r in rows:
        b = jload(r["bbox"], [])
        if len(b) == 4:
            by_cam.setdefault(r["camera_id"], []).append((r, b))
    out: set[int] = set()
    for items in by_cam.values():
        B = np.array([b for _, b in items], dtype=np.float32)
        T = np.array([r["first_ts"] for r, _ in items])
        area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
        for k, (r, b) in enumerate(items):
            if r["moving"] is not None:
                continue
            iw = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
            ih = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
            inter = iw * ih
            iou = inter / np.maximum(area[k] + area - inter, 1e-9)
            if int(((iou >= PARKED_IOU) & (np.abs(T - T[k]) > gap_s)).sum()) >= PARKED_REPEATS:
                out.add(r["id"])
    return out


def _close_plates(a: str, b: str) -> bool:
    """Same plate allowing one OCR slip (substitution, or one character missing)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))
