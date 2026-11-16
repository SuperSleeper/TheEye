"""Named people (family members): enrolment, face matching, tags, and "is this ...?" suggestions.

faces.status: unknown (no match yet) | auto (matched a person, not confirmed) | confirmed (you said so)
| rejected (you said it's not that person; faces.rejected lists who it isn't).
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
from .faces import MATCH, SUGGEST, FaceEngine, similarity

log = logging.getLogger("theeye.people")

SCAN_IDLE_LOAD = 0.5          # past-footage scan pauses while the load average is above this share of the CPUs
SCAN_FRAMES = 6               # moments per person track decoded from the clip when scanning past footage
ENROL_FRAMES = 10             # ...and when you mark someone yourself: worth looking harder


class People:
    def __init__(self, hub, models_dir: Path):
        self.hub, self.db = hub, hub.db
        self.engine = FaceEngine(models_dir)
        self._lock = threading.RLock()
        self._refs: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}
        self.stats = {"scanned": 0, "faces": 0, "scan_left": None}
        self.reload()

    # ------------------------------------------------------------ registry
    def reload(self) -> None:
        with self._lock:
            self._names = {r["id"]: r["name"] for r in self.db.all("SELECT id, name FROM people")}
            refs: dict[int, list] = {}
            for r in self.db.all("SELECT person_id, embedding FROM faces WHERE status='confirmed'"):
                refs.setdefault(r["person_id"], []).append(np.frombuffer(r["embedding"], dtype=np.float32))
            self._refs = {p: np.stack(v) for p, v in refs.items()}

    def names(self) -> dict[int, str]:
        return dict(self._names)

    def add_person(self, name: str) -> int:
        pid = self.db.insert("INSERT INTO people(name, created) VALUES(?,?)", (name.strip(), time.time()))
        self.reload()
        return pid

    def remove_person(self, pid: int) -> None:
        name = self._names.get(pid)
        dets = [r["detection_id"] for r in self.db.all("SELECT detection_id FROM faces WHERE person_id=?", (pid,))]
        self.db.execute("UPDATE faces SET person_id=NULL, status='unknown' WHERE person_id=?", (pid,))
        self.db.execute("DELETE FROM people WHERE id=?", (pid,))
        self.reload()
        for d in dets:
            self._set_who(d, None, old=name)

    # ------------------------------------------------------------ faces
    def add_face(self, det_id: int, cam: str, ts: float, img, bbox: list[float]) -> Optional[int]:
        f = self.engine.best_face(img, bbox)
        if f is None:
            return None
        feat, aligned, size = f
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        rel = Path("snaps") / "faces" / cam / day / f"face_{det_id}.jpg"
        out = self.hub.data / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.db.execute("DELETE FROM faces WHERE detection_id=? AND status IN ('unknown','auto')", (det_id,))
        fid = self.db.insert("INSERT INTO faces(detection_id, camera_id, ts, size, embedding, path, status)"
                             " VALUES(?,?,?,?,?,?,'unknown')", (det_id, cam, ts, size, feat.tobytes(), str(rel)))
        self.stats["faces"] += 1
        self._match(self.db.one("SELECT * FROM faces WHERE id=?", (fid,)))
        return fid

    def _best_person(self, feat: np.ndarray, rejected: set) -> tuple[Optional[int], float]:
        best, best_s = None, 0.0
        with self._lock:
            for pid, refs in self._refs.items():
                if pid in rejected:
                    continue
                s = similarity(feat, refs)
                if s > best_s:
                    best, best_s = pid, s
        return best, best_s

    def _match(self, row) -> None:
        """Tag (or untag) one unconfirmed face with the person it matches."""
        if row is None or row["status"] in ("confirmed", "rejected"):
            return
        feat = np.frombuffer(row["embedding"], dtype=np.float32)
        pid, s = self._best_person(feat, set(jload(row["rejected"], [])))
        if pid is not None and s >= MATCH:
            self.db.execute("UPDATE faces SET person_id=?, status='auto', score=? WHERE id=?", (pid, s, row["id"]))
            self._set_who(row["detection_id"], self._names.get(pid))
        elif row["status"] == "auto":
            self.db.execute("UPDATE faces SET person_id=NULL, status='unknown', score=? WHERE id=?", (s, row["id"]))
            self._set_who(row["detection_id"], None, old=self._names.get(row["person_id"]))
        else:
            self.db.execute("UPDATE faces SET score=? WHERE id=?", (s, row["id"]))

    def rematch_all(self) -> None:
        for row in self.db.all("SELECT * FROM faces WHERE status IN ('unknown','auto')"):
            self._match(row)

    def confirm(self, face_id: int, pid: int) -> None:
        row = self.db.one("SELECT * FROM faces WHERE id=?", (face_id,))
        if not row or pid not in self._names:
            raise KeyError(face_id)
        old = self._names.get(row["person_id"]) if row["person_id"] != pid else None
        self.db.execute("UPDATE faces SET person_id=?, status='confirmed' WHERE id=?", (pid, face_id))
        self._set_who(row["detection_id"], self._names[pid], old=old)
        self.reload()
        threading.Thread(target=self.rematch_all, name="faces-rematch", daemon=True).start()

    def reject(self, face_id: int, pid: int) -> None:
        row = self.db.one("SELECT * FROM faces WHERE id=?", (face_id,))
        if not row:
            raise KeyError(face_id)
        rejected = sorted(set(jload(row["rejected"], [])) | {pid})
        was = row["person_id"] == pid
        self.db.execute("UPDATE faces SET rejected=?, status=?, person_id=? WHERE id=?",
                        (json.dumps(rejected), "unknown" if was else row["status"],
                         None if was else row["person_id"], face_id))
        if was:
            self._set_who(row["detection_id"], None, old=self._names.get(pid))
            self.reload()
            threading.Thread(target=self.rematch_all, name="faces-rematch", daemon=True).start()
        self._match(self.db.one("SELECT * FROM faces WHERE id=?", (face_id,)))

    def enrol_detection(self, det_id: int, pid: int) -> Optional[int]:
        """'This is <person>' on a detection: use its face (finding it in the recording if needed)."""
        row = self.db.one("SELECT id FROM faces WHERE detection_id=? ORDER BY size DESC LIMIT 1", (det_id,))
        fid = row["id"] if row else self._scan_detection(det_id, ENROL_FRAMES)
        if fid:
            self.confirm(fid, pid)
        return fid

    def suggestions(self, pid: int, limit: int = 60) -> list[dict]:
        with self._lock:
            refs = self._refs.get(pid)
        if refs is None:
            return []
        out = []
        for r in self.db.all("SELECT id, detection_id, camera_id, ts, size, embedding, path, status, person_id, rejected"
                             " FROM faces WHERE status IN ('unknown','auto')"):
            if pid in jload(r["rejected"], []) or (r["status"] == "auto" and r["person_id"] != pid):
                continue
            s = similarity(np.frombuffer(r["embedding"], dtype=np.float32), refs)
            if s >= SUGGEST:
                out.append({k: r[k] for k in ("id", "detection_id", "camera_id", "ts", "size", "path", "status")}
                           | {"score": round(s, 3)})
        out.sort(key=lambda x: -x["score"])
        return out[:limit]

    def confirmed(self, pid: int) -> list[dict]:
        return self.db.all("SELECT id, detection_id, camera_id, ts, size, path FROM faces"
                           " WHERE person_id=? AND status='confirmed' ORDER BY ts DESC", (pid,))

    # ------------------------------------------------------------ tags
    def _set_who(self, det_id: int, name: Optional[str], old: Optional[str] = None) -> None:
        """Name a detection (or clear it), and keep its recordings' tags in step."""
        self.db.execute("UPDATE detections SET who=? WHERE id=?", (name, det_id))
        for r in self.db.all("SELECT event_id FROM event_links WHERE kind='detection' AND ref_id=?", (det_id,)):
            eid = r["event_id"]
            others = {x["who"] for x in self.db.all(
                "SELECT d.who FROM event_links l JOIN detections d ON d.id=l.ref_id AND l.kind='detection'"
                " WHERE l.event_id=? AND d.who IS NOT NULL", (eid,))}
            with self.hub._lock:
                ev = next((e for e in self.hub._open_events.values() if e.id == eid), None)
                labels = set(ev.labels) if ev else set(jload((self.db.one("SELECT labels FROM events WHERE id=?",
                                                                          (eid,)) or {}).get("labels"), []))
                if name:
                    labels.add(name)
                if old and old != name and old not in others:
                    labels.discard(old)
                if ev:
                    ev.labels = labels
                    self.hub._flush_event_meta(ev)
                else:
                    self.db.execute("UPDATE events SET labels=? WHERE id=?", (json.dumps(sorted(labels)), eid))

    # ------------------------------------------------------------ live and past footage
    def on_track_end(self, t) -> None:
        if t.label != "person" or t.db_id is None:
            return
        cam = self.hub.cams.get(t.camera_id)
        frame = cam.decode_frame_at(t.best_ts) if cam and t.best_ts else None       # full resolution
        if frame is None:
            frame = t.best_frame
        try:
            self.add_face(t.db_id, t.camera_id, t.best_ts or t.last_ts, frame, t.best_bbox)
        finally:
            self.db.execute("UPDATE detections SET face_checked=1 WHERE id=?", (t.db_id,))

    def _scan_detection(self, det_id: int, frames: int = SCAN_FRAMES) -> Optional[int]:
        import av
        r = self.db.one("SELECT d.first_ts, d.last_ts, d.bbox, d.camera_id, e.clip_path, e.clip_offset, e.start_ts"
                        " FROM detections d JOIN events e ON e.id = d.event_id"
                        " WHERE d.id=? AND e.clip_path IS NOT NULL", (det_id,))
        self.db.execute("UPDATE detections SET face_checked=1 WHERE id=?", (det_id,))
        if not r or not (self.hub.data / r["clip_path"]).exists():
            return None
        bbox = jload(r["bbox"], [])
        last = max(r["last_ts"] or r["first_ts"], r["first_ts"] + 1)
        a, z = (r["clip_offset"] + t - r["start_ts"] for t in (r["first_ts"], last))
        want = list(np.linspace(a, z, frames))
        best = None
        try:
            with av.open(str(self.hub.data / r["clip_path"])) as ct:
                vs = ct.streams.video[0]
                ct.seek(int(max(0.0, a - 1) / vs.time_base), stream=vs, backward=True)
                for fr in ct.decode(vs):
                    t = float(fr.pts * vs.time_base) if fr.pts is not None else 0.0
                    if want and t >= want[0]:
                        want.pop(0)
                        # the stored box is from the best moment; other moments get a wider search area
                        wide = [bbox[0] - 0.15, bbox[1] - 0.1, bbox[2] + 0.15, bbox[3] + 0.05] if len(bbox) == 4 else bbox
                        f = self.engine.best_face(fr.to_ndarray(format="bgr24"), wide)
                        if f and (best is None or f[2] > best[0][2]):
                            best = (f, t)
                    if not want:
                        break
        except Exception:  # noqa: BLE001
            log.debug("face scan of detection %d failed", det_id, exc_info=True)
            return None
        if best is None:
            return None
        ts = r["start_ts"] + best[1] - r["clip_offset"]      # the moment with the largest face
        return self._store_scanned(det_id, r["camera_id"], ts, best[0])

    def _store_scanned(self, det_id: int, cam: str, ts: float, f) -> int:
        feat, aligned, size = f
        day = time.strftime("%Y-%m-%d", time.localtime(ts))
        rel = Path("snaps") / "faces" / cam / day / f"face_{det_id}.jpg"
        (self.hub.data / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.hub.data / rel), aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
        fid = self.db.insert("INSERT INTO faces(detection_id, camera_id, ts, size, embedding, path, status)"
                             " VALUES(?,?,?,?,?,?,'unknown')", (det_id, cam, ts, size, feat.tobytes(), str(rel)))
        self.stats["faces"] += 1
        self._match(self.db.one("SELECT * FROM faces WHERE id=?", (fid,)))
        return fid

    def scan_past(self) -> None:
        """Background: find faces in past recordings (newest first) while the machine is quiet."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (AttributeError, OSError):
            pass
        stop = self.hub._stop
        q = (" FROM detections d JOIN events e ON e.id = d.event_id WHERE d.label='person' AND d.face_checked IS NULL"
             " AND e.clip_path IS NOT NULL AND e.end_ts IS NOT NULL")
        while not stop.wait(2):
            if not self.hub.cfg.people.enabled or not self.engine._load():
                stop.wait(300)
                continue
            if os.getloadavg()[0] > SCAN_IDLE_LOAD * (os.cpu_count() or 4):
                stop.wait(60)
                continue
            rows = self.db.all("SELECT d.id" + q + " ORDER BY d.first_ts DESC LIMIT 20")
            self.stats["scan_left"] = self.db.one("SELECT COUNT(*) AS n" + q)["n"]
            if not rows:
                stop.wait(600)
                continue
            for r in rows:
                if stop.is_set():
                    return
                self._scan_detection(r["id"])
                self.stats["scanned"] += 1
