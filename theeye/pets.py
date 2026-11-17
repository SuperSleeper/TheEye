"""Named pets: enrolment, matching by look, tags, and "is this ...?" suggestions. Same flow as theeye.people.

A pet's "look" is a CLIP picture of just the animal: the inside of its box in the snapshot. (The detection's own
embedding includes a margin around the box, and for a small dog that's mostly patio, so every dog in the same yard
looked alike.) pet_looks.status: unknown | auto (matched, not confirmed) | confirmed | rejected, as for faces.
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

log = logging.getLogger("theeye.pets")

PET_LABELS = ("dog", "cat")      # detector labels a pet can come out as (a fluffy dog is sometimes a "cat")
MATCH = 0.82                     # tagged automatically from this similarity to a pet's confirmed looks...
MARGIN = 0.03                    # ...when it's also this much closer to that pet than to any other
SUGGEST = 0.70                   # offered as "is this ...?" from here
TOP_K = 3                        # a look's score: mean of its best matches among a pet's confirmed looks
MIN_PX = 24                      # smaller animals are too blurry to tell apart
INSET = 0.08                     # crop this share in from each side of the box: the snapshot's box line is drawn there
SCAN_IDLE_LOAD = 0.5


class Pets:
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
            self._names = {r["id"]: r["name"] for r in self.db.all("SELECT id, name FROM pets")}
            refs: dict[int, list] = {}
            for r in self.db.all("SELECT pet_id, embedding FROM pet_looks WHERE status='confirmed'"):
                refs.setdefault(r["pet_id"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
            self._refs = {p: np.stack(v) for p, v in refs.items()}

    def names(self) -> dict[int, str]:
        return dict(self._names)

    def add_pet(self, name: str) -> int:
        pid = self.db.insert("INSERT INTO pets(name, created) VALUES(?,?)", (name.strip(), time.time()))
        self.reload()
        return pid

    def remove_pet(self, pid: int) -> None:
        name = self._names.get(pid)
        dets = [r["detection_id"] for r in self.db.all("SELECT detection_id FROM pet_looks WHERE pet_id=?", (pid,))]
        self.db.execute("UPDATE pet_looks SET pet_id=NULL, status='unknown' WHERE pet_id=?", (pid,))
        self.db.execute("DELETE FROM pets WHERE id=?", (pid,))
        self.reload()
        for d in dets:
            set_who(self.hub, d, None, old=name)

    # ------------------------------------------------------------ looks
    def add_look(self, det_id: int) -> Optional[int]:
        """Picture the animal of one detection (from its snapshot) and match it. Returns the look's id."""
        r = self.db.one("SELECT camera_id, first_ts, bbox, snapshot_path FROM detections WHERE id=?", (det_id,))
        emb = self.hub.embedder
        if not r or not r["snapshot_path"] or emb is None:
            return None
        img = cv2.imread(str(self.hub.data / r["snapshot_path"]))
        b = jload(r["bbox"], [])
        if img is None or len(b) != 4:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        size = int(max(x2 - x1, y2 - y1))
        if size < MIN_PX:
            return None
        dx, dy = (x2 - x1) * INSET, (y2 - y1) * INSET
        crop = img[int(y1 + dy):int(y2 - dy), int(x1 + dx):int(x2 - dx)]
        if crop.size == 0:
            return None
        feat = emb.image(crop)
        if feat is None:
            return None
        day = time.strftime("%Y-%m-%d", time.localtime(r["first_ts"]))
        rel = Path("snaps") / "faces" / r["camera_id"] / day / f"pet_{det_id}.jpg"
        (self.hub.data / rel).parent.mkdir(parents=True, exist_ok=True)
        # a square around the animal, with some room, for the Pets page
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, size * 1.3
        sq = img[max(0, int(cy - s / 2)):int(cy + s / 2), max(0, int(cx - s / 2)):int(cx + s / 2)]
        cv2.imwrite(str(self.hub.data / rel), cv2.resize(sq, (160, 160), interpolation=cv2.INTER_AREA)
                    if sq.size else crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.db.execute("DELETE FROM pet_looks WHERE detection_id=? AND status IN ('unknown','auto')", (det_id,))
        if self.db.one("SELECT 1 FROM pet_looks WHERE detection_id=?", (det_id,)):
            return None                                # already confirmed/rejected: keep the verdict
        lid = self.db.insert("INSERT INTO pet_looks(detection_id, camera_id, ts, size, embedding, path, status)"
                             " VALUES(?,?,?,?,?,?,'unknown')",
                             (det_id, r["camera_id"], r["first_ts"], size, feat.tobytes(), str(rel)))
        self.stats["looks"] += 1
        self._match(self.db.one("SELECT * FROM pet_looks WHERE id=?", (lid,)))
        return lid

    def _scores(self, feat: np.ndarray, rejected: set) -> dict[int, float]:
        out = {}
        with self._lock:
            for pid, refs in self._refs.items():
                if pid in rejected:
                    continue
                s = np.sort(refs @ feat)[-TOP_K:]
                out[pid] = float(s.mean())
        return out

    def _match(self, row) -> None:
        """Tag (or untag) one unconfirmed look with the pet it matches."""
        if row is None or row["status"] in ("confirmed", "rejected"):
            return
        feat = np.frombuffer(row["embedding"], dtype=np.float32)
        sc = self._scores(feat, set(jload(row["rejected"], [])))
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        pid, s = ranked[0] if ranked else (None, 0.0)
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if pid is not None and s >= MATCH and s - runner_up >= MARGIN:
            old = self._names.get(row["pet_id"]) if row["pet_id"] != pid else None
            self.db.execute("UPDATE pet_looks SET pet_id=?, status='auto', score=? WHERE id=?", (pid, s, row["id"]))
            set_who(self.hub, row["detection_id"], self._names.get(pid), old=old)
        elif row["status"] == "auto":
            self.db.execute("UPDATE pet_looks SET pet_id=NULL, status='unknown', score=? WHERE id=?", (s, row["id"]))
            set_who(self.hub, row["detection_id"], None, old=self._names.get(row["pet_id"]))
        else:
            self.db.execute("UPDATE pet_looks SET score=? WHERE id=?", (s, row["id"]))

    def rematch_all(self) -> None:
        for row in self.db.all("SELECT * FROM pet_looks WHERE status IN ('unknown','auto')"):
            self._match(row)

    def _rematch_soon(self) -> None:
        threading.Thread(target=self.rematch_all, name="pets-rematch", daemon=True).start()

    def confirm(self, look_id: int, pid: int) -> None:
        row = self.db.one("SELECT * FROM pet_looks WHERE id=?", (look_id,))
        if not row or pid not in self._names:
            raise KeyError(look_id)
        old = self._names.get(row["pet_id"]) if row["pet_id"] != pid else None
        self.db.execute("UPDATE pet_looks SET pet_id=?, status='confirmed' WHERE id=?", (pid, look_id))
        set_who(self.hub, row["detection_id"], self._names[pid], old=old)
        self.reload()
        self._rematch_soon()

    def reject(self, look_id: int, pid: int) -> None:
        row = self.db.one("SELECT * FROM pet_looks WHERE id=?", (look_id,))
        if not row:
            raise KeyError(look_id)
        rejected = sorted(set(jload(row["rejected"], [])) | {pid})
        was = row["pet_id"] == pid
        self.db.execute("UPDATE pet_looks SET rejected=?, status=?, pet_id=? WHERE id=?",
                        (json.dumps(rejected), "unknown" if was else row["status"],
                         None if was else row["pet_id"], look_id))
        if was:
            set_who(self.hub, row["detection_id"], None, old=self._names.get(pid))
            if row["status"] == "confirmed":
                self.reload()
                self._rematch_soon()
        self._match(self.db.one("SELECT * FROM pet_looks WHERE id=?", (look_id,)))

    def enrol_detection(self, det_id: int, pid: int) -> Optional[int]:
        """'This is <pet>' on a detection."""
        row = self.db.one("SELECT id FROM pet_looks WHERE detection_id=?", (det_id,))
        lid = row["id"] if row else self.add_look(det_id)
        if lid:
            self.confirm(lid, pid)
        return lid

    def unmark_detection(self, det_id: int) -> int:
        row = self.db.one("SELECT id, pet_id FROM pet_looks WHERE detection_id=? AND pet_id IS NOT NULL", (det_id,))
        if row:
            self.reject(row["id"], row["pet_id"])
            return 1
        d = self.db.one("SELECT who FROM detections WHERE id=?", (det_id,))
        if d and d["who"] in self._names.values():
            set_who(self.hub, det_id, None, old=d["who"])
            return 1
        return 0

    def suggestions(self, pid: int, limit: int = 60) -> list[dict]:
        with self._lock:
            refs = self._refs.get(pid)
        if refs is None:              # nothing to compare with yet: offer the clearest dogs and cats seen to start from
            rows = [r for r in self.db.all("SELECT id, detection_id, camera_id, ts, size, path, status, rejected"
                                           " FROM pet_looks WHERE status='unknown' ORDER BY size DESC, ts DESC")
                    if pid not in jload(r["rejected"], [])][:limit]
            return self._with_clip([{k: r[k] for k in ("id", "detection_id", "camera_id", "ts", "size", "path",
                                                       "status")} | {"score": None} for r in rows])
        out = []
        for r in self.db.all("SELECT id, detection_id, camera_id, ts, size, embedding, path, status, pet_id, rejected"
                             " FROM pet_looks WHERE status IN ('unknown','auto')"):
            if pid in jload(r["rejected"], []) or (r["status"] == "auto" and r["pet_id"] != pid):
                continue
            s = float(np.sort(refs @ np.frombuffer(r["embedding"], dtype=np.float32))[-TOP_K:].mean())
            if s >= SUGGEST:
                out.append({k: r[k] for k in ("id", "detection_id", "camera_id", "ts", "size", "path", "status")}
                           | {"score": round(s, 3)})
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
        return self._with_clip(self.db.all("SELECT id, detection_id, camera_id, ts, size, path FROM pet_looks"
                                           " WHERE pet_id=? AND status='confirmed' ORDER BY ts DESC", (pid,)))

    # ------------------------------------------------------------ live and past sightings
    def on_track_end(self, t) -> None:
        if t.label in PET_LABELS and t.db_id is not None:
            self.add_look(t.db_id)

    def scan_past(self) -> None:
        """Background: picture past dogs and cats (newest first) while the machine is quiet."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (AttributeError, OSError):
            pass
        stop = self.hub._stop
        q = ("SELECT d.id FROM detections d LEFT JOIN pet_looks p ON p.detection_id = d.id"
             f" WHERE d.label IN ({','.join('?' * len(PET_LABELS))}) AND p.id IS NULL AND d.snapshot_path IS NOT NULL"
             " ORDER BY d.first_ts DESC")
        skip: set[int] = set()            # too small or no snapshot file: don't retry this run
        while not stop.wait(5):
            if self.hub.embedder is None:
                return
            if os.getloadavg()[0] > SCAN_IDLE_LOAD * (os.cpu_count() or 4):
                stop.wait(60)
                continue
            todo = [r["id"] for r in self.db.all(q, PET_LABELS) if r["id"] not in skip]
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
                    log.debug("pet look for detection %d failed", did, exc_info=True)
                    skip.add(did)
