"""Things you name that are recognised by how they look: pets (theeye.pets) and your vehicles (theeye.cars).

Same flow as faces (theeye.people): you name one, confirm pictures of it, and later sightings that look like those
pictures get its name. A "look" is a CLIP picture of just the object: the inside of its box in the snapshot (the
detection's own embedding includes a margin around the box, and for a small dog that's mostly patio).
<looks>.status: unknown | auto (matched, not confirmed) | confirmed | rejected, as for faces.

A subclass sets the tables, the detector labels it applies to and its thresholds.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .db import jload
from .people import set_who

log = logging.getLogger("theeye.looks")

INSET = 0.08                     # crop this share in from each side of the box: the snapshot's box line is drawn there
SCAN_IDLE_LOAD = 0.5


class NamedLooks:
    REG = "pets"                 # table of named things (id, name, cover, ...)
    LOOKS = "pet_looks"          # table of their pictures
    OWNER = "pet_id"             # column in LOOKS naming the thing
    LABELS: tuple = ("dog", "cat")
    PREFIX = "pet"               # picture file name prefix
    MATCH = 0.82                 # tagged automatically from this similarity to its confirmed looks...
    MARGIN = 0.03                # ...when it's also this much closer to it than to any other
    SUGGEST = 0.70               # offered as "is this ...?" from here
    TOP_K = 3                    # a look's score: mean of its best matches among the confirmed looks
    MIN_PX = 24                  # smaller ones are too blurry to tell apart
    ROOM = 1.3                   # the page's square picture: this much of the box's longer side
    FILTER = ""                  # SQL on LOOKS rows that may be suggested at all (" AND ...")
    SCAN_WHERE = ""              # SQL on detections d that are worth picturing in the past scan (" AND ...")

    def __init__(self, hub):
        self.hub, self.db = hub, hub.db
        self._lock = threading.RLock()
        self._refs: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}
        self.stats = {"looks": 0, "scan_left": None}
        self.reload()

    # ------------------------------------------------------------ registry
    def reload(self) -> None:
        with self._lock:
            self._names = {r["id"]: r["name"] for r in self.db.all(f"SELECT id, name FROM {self.REG}")}
            refs: dict[int, list] = {}
            for r in self.db.all(f"SELECT {self.OWNER} AS o, embedding FROM {self.LOOKS} WHERE status='confirmed'"):
                refs.setdefault(r["o"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
            self._refs = {p: np.stack(v) for p, v in refs.items()}

    def names(self) -> dict[int, str]:
        return dict(self._names)

    def add(self, name: str) -> int:
        pid = self.db.insert(f"INSERT INTO {self.REG}(name, created) VALUES(?,?)", (name.strip(), time.time()))
        self.reload()
        return pid

    def remove(self, pid: int) -> None:
        name = self._names.get(pid)
        dets = [r["detection_id"] for r in self.db.all(f"SELECT detection_id FROM {self.LOOKS} WHERE {self.OWNER}=?",
                                                       (pid,))]
        self.db.execute(f"UPDATE {self.LOOKS} SET {self.OWNER}=NULL, status='unknown' WHERE {self.OWNER}=?", (pid,))
        self.db.execute(f"DELETE FROM {self.REG} WHERE id=?", (pid,))
        self.reload()
        for d in dets:
            set_who(self.hub, d, None, old=name)

    # ------------------------------------------------------------ looks
    def add_look(self, det_id: int) -> Optional[int]:
        """Picture the object of one detection (from its snapshot) and match it. Returns the look's id."""
        look = tight_look(self.hub, det_id, self.MIN_PX)
        if look is None:
            return None
        feat, size, img, (x1, y1, x2, y2), crop = look
        r = self.db.one("SELECT camera_id, first_ts FROM detections WHERE id=?", (det_id,))
        day = time.strftime("%Y-%m-%d", time.localtime(r["first_ts"]))
        rel = Path("snaps") / "faces" / r["camera_id"] / day / f"{self.PREFIX}_{det_id}.jpg"
        (self.hub.data / rel).parent.mkdir(parents=True, exist_ok=True)
        # a square around it, with some room, for the People & Pets page
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, size * self.ROOM
        sq = img[max(0, int(cy - s / 2)):int(cy + s / 2), max(0, int(cx - s / 2)):int(cx + s / 2)]
        cv2.imwrite(str(self.hub.data / rel), cv2.resize(sq, (160, 160), interpolation=cv2.INTER_AREA)
                    if sq.size else crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.db.execute(f"DELETE FROM {self.LOOKS} WHERE detection_id=? AND status IN ('unknown','auto')", (det_id,))
        if self.db.one(f"SELECT 1 FROM {self.LOOKS} WHERE detection_id=?", (det_id,)):
            return None                                # already confirmed/rejected: keep the verdict
        lid = self.db.insert(f"INSERT INTO {self.LOOKS}(detection_id, camera_id, ts, size, embedding, path, status)"
                             " VALUES(?,?,?,?,?,?,'unknown')",
                             (det_id, r["camera_id"], r["first_ts"], size, feat.tobytes(), str(rel)))
        self.stats["looks"] += 1
        self._match(self._row(lid))
        return lid

    def _row(self, lid: int):
        return self.db.one(f"SELECT *, {self.OWNER} AS owner FROM {self.LOOKS} WHERE id=?", (lid,))

    def _scores(self, feat: np.ndarray, rejected: set) -> dict[int, float]:
        out = {}
        with self._lock:
            for pid, refs in self._refs.items():
                if pid in rejected:
                    continue
                s = np.sort(refs @ feat)[-self.TOP_K:]
                out[pid] = float(s.mean())
        return out

    def _pick(self, row, feat: np.ndarray) -> tuple[Optional[int], float]:
        """Which one this look is (or None), and its best score."""
        sc = self._scores(feat, set(jload(row["rejected"], [])))
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        pid, s = ranked[0] if ranked else (None, 0.0)
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return (pid if pid is not None and s >= self.MATCH and s - runner_up >= self.MARGIN else None), s

    def _match(self, row) -> None:
        """Tag (or untag) one unconfirmed look with the one it matches."""
        if row is None or row["status"] in ("confirmed", "rejected"):
            return
        pid, s = self._pick(row, np.frombuffer(row["embedding"], dtype=np.float32))
        if pid is not None:
            old = self._names.get(row["owner"]) if row["owner"] != pid else None
            self.db.execute(f"UPDATE {self.LOOKS} SET {self.OWNER}=?, status='auto', score=? WHERE id=?",
                            (pid, s, row["id"]))
            set_who(self.hub, row["detection_id"], self._names.get(pid), old=old)
        elif row["status"] == "auto":
            self.db.execute(f"UPDATE {self.LOOKS} SET {self.OWNER}=NULL, status='unknown', score=? WHERE id=?",
                            (s, row["id"]))
            set_who(self.hub, row["detection_id"], None, old=self._names.get(row["owner"]))
        else:
            self.db.execute(f"UPDATE {self.LOOKS} SET score=? WHERE id=?", (s, row["id"]))

    def rematch_all(self) -> None:
        for row in self.db.all(f"SELECT *, {self.OWNER} AS owner FROM {self.LOOKS} WHERE status IN ('unknown','auto')"):
            self._match(row)

    def _rematch_soon(self) -> None:
        threading.Thread(target=self.rematch_all, name=f"{self.PREFIX}-rematch", daemon=True).start()

    def confirm(self, look_id: int, pid: int) -> None:
        row = self._row(look_id)
        if not row or pid not in self._names:
            raise KeyError(look_id)
        old = self._names.get(row["owner"]) if row["owner"] != pid else None
        self.db.execute(f"UPDATE {self.LOOKS} SET {self.OWNER}=?, status='confirmed' WHERE id=?", (pid, look_id))
        set_who(self.hub, row["detection_id"], self._names[pid], old=old)
        self.reload()
        self.on_confirm(row, pid)
        self._rematch_soon()

    def on_confirm(self, row, pid: int) -> None:
        """Hook: a look was just confirmed as pid."""

    def reject(self, look_id: int, pid: int) -> None:
        row = self._row(look_id)
        if not row:
            raise KeyError(look_id)
        rejected = sorted(set(jload(row["rejected"], [])) | {pid})
        was = row["owner"] == pid
        self.db.execute(f"UPDATE {self.LOOKS} SET rejected=?, status=?, {self.OWNER}=? WHERE id=?",
                        (json.dumps(rejected), "unknown" if was else row["status"],
                         None if was else row["owner"], look_id))
        if was:
            set_who(self.hub, row["detection_id"], None, old=self._names.get(pid))
            if row["status"] == "confirmed":
                self.reload()
                self._rematch_soon()
        self._match(self._row(look_id))

    def enrol_detection(self, det_id: int, pid: int) -> Optional[int]:
        """'This is <name>' on a detection."""
        row = self.db.one(f"SELECT id FROM {self.LOOKS} WHERE detection_id=?", (det_id,))
        lid = row["id"] if row else self.add_look(det_id)
        if lid:
            self.confirm(lid, pid)
        return lid

    def unmark_detection(self, det_id: int) -> int:
        row = self.db.one(f"SELECT id, {self.OWNER} AS owner FROM {self.LOOKS}"
                          f" WHERE detection_id=? AND {self.OWNER} IS NOT NULL", (det_id,))
        if row:
            self.reject(row["id"], row["owner"])
            return 1
        d = self.db.one("SELECT who FROM detections WHERE id=?", (det_id,))
        if d and d["who"] in self._names.values():
            set_who(self.hub, det_id, None, old=d["who"])
            return 1
        return 0

    def suggestions(self, pid: int, limit: int = 60) -> list[dict]:
        cols = ("id", "detection_id", "camera_id", "ts", "size", "path", "status")
        with self._lock:
            refs = self._refs.get(pid)
        if refs is None:              # nothing to compare with yet: what's tagged already (a vehicle's plate), then
            rows = [r for r in self.db.all(        # the clearest ones seen to start from
                "SELECT id, detection_id, camera_id, ts, size, path, status, rejected, score"
                f" FROM {self.LOOKS} WHERE (status='unknown' OR (status='auto' AND {self.OWNER}=?))" + self.FILTER +
                " ORDER BY status='auto' DESC, size DESC, ts DESC LIMIT ?", (pid, limit * 3))
                    if pid not in jload(r["rejected"], [])][:limit]
            return self._with_clip([{k: r[k] for k in cols} | {"score": r["score"] if r["status"] == "auto" else None}
                                    for r in rows])
        out = []
        for r in self.db.all("SELECT id, detection_id, camera_id, ts, size, embedding, path, status, score,"
                             f" {self.OWNER} AS owner, rejected FROM {self.LOOKS}"
                             " WHERE status IN ('unknown','auto')" + self.FILTER):
            if pid in jload(r["rejected"], []) or (r["status"] == "auto" and r["owner"] != pid):
                continue
            s = float(np.sort(refs @ np.frombuffer(r["embedding"], dtype=np.float32))[-self.TOP_K:].mean())
            if r["status"] == "auto":            # tagged as this one (by look, or by a vehicle's plate)
                s = max(s, r["score"] or 0.0)
            if s >= self.SUGGEST or r["status"] == "auto":
                out.append({k: r[k] for k in cols} | {"score": round(s, 3)})
        # unconfirmed tags first (the ones worth checking), then the closest
        out.sort(key=lambda x: (x["status"] != "auto", -x["score"]))
        return self._with_clip(out[:limit])

    def _with_clip(self, looks: list[dict]) -> list[dict]:
        for f in looks:
            e = self.db.one("SELECT e.id, e.start_ts, e.clip_offset FROM event_links l JOIN events e ON e.id=l.event_id"
                            " WHERE l.kind='detection' AND l.ref_id=? AND e.clip_path IS NOT NULL"
                            " ORDER BY e.start_ts LIMIT 1", (f["detection_id"],))
            f["event_id"] = e["id"] if e else None
            f["seek"] = max(0.0, e["clip_offset"] + f["ts"] - e["start_ts"] - 2) if e else None
        return looks

    def confirmed(self, pid: int) -> list[dict]:
        return self._with_clip(self.db.all(f"SELECT id, detection_id, camera_id, ts, size, path FROM {self.LOOKS}"
                                           f" WHERE {self.OWNER}=? AND status='confirmed' ORDER BY ts DESC", (pid,)))

    # ------------------------------------------------------------ live and past sightings
    def wants(self, t) -> bool:
        return t.label in self.LABELS

    def on_track_end(self, t) -> None:
        if t.db_id is not None and self.wants(t):
            self.add_look(t.db_id)

    def scan_past(self) -> None:
        """Background: picture past sightings (newest first) while the machine is quiet."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (AttributeError, OSError):
            pass
        stop = self.hub._stop
        q = (f"SELECT d.id FROM detections d LEFT JOIN {self.LOOKS} p ON p.detection_id = d.id"
             f" WHERE d.label IN ({','.join('?' * len(self.LABELS))}) AND p.id IS NULL AND d.snapshot_path IS NOT NULL"
             + self.SCAN_WHERE + " ORDER BY d.first_ts DESC")
        skip: set[int] = set()            # too small or no snapshot file: don't retry this run
        while not stop.wait(5):
            if self.hub.embedder is None:
                return
            if os.getloadavg()[0] > SCAN_IDLE_LOAD * (os.cpu_count() or 4):
                stop.wait(60)
                continue
            todo = [r["id"] for r in self.db.all(q, self.LABELS) if r["id"] not in skip]
            self.stats["scan_left"] = len(todo)
            rows = todo[:20]
            if not rows:
                stop.wait(600)
                continue
            for did in rows:
                if stop.is_set():
                    return
                try:
                    if not self.add_look(did):
                        skip.add(did)
                except Exception:  # noqa: BLE001
                    log.debug("%s look for detection %d failed", self.PREFIX, did, exc_info=True)
                    skip.add(did)


def tight_look(hub, det_id: int, min_px: int = 24):
    """(CLIP picture of just the object, its size in px, the snapshot, its box in px, the crop), or None when there's
    no snapshot or it's too small to tell apart."""
    r = hub.db.one("SELECT bbox, snapshot_path FROM detections WHERE id=?", (det_id,))
    emb = hub.embedder
    if not r or not r["snapshot_path"] or emb is None:
        return None
    img = cv2.imread(str(hub.data / r["snapshot_path"]))
    b = jload(r["bbox"], [])
    if img is None or len(b) != 4:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
    size = int(max(x2 - x1, y2 - y1))
    if size < min_px:
        return None
    dx, dy = (x2 - x1) * INSET, (y2 - y1) * INSET
    crop = img[int(y1 + dy):int(y2 - dy), int(x1 + dx):int(x2 - dx)]
    if crop.size == 0:
        return None
    feat = emb.image(crop)
    if feat is None:
        return None
    return feat, size, img, (x1, y1, x2, y2), crop
