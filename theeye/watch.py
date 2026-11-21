"""Watch zones: is something that stays put (trash cans at the curb, a gate, a bike) there or not?

The object detector only knows moving-object classes (no "trash can"), and motion only says something changed, so a
watch zone is checked as a picture instead: once a minute its crop of the main stream is compared (CLIP) with crops
you marked "there", "gone" or "can't see" (a car parked in front). Plain-text CLIP prompts were tried first and found
the cans in 1 of 4 crops; examples of this exact spot separate them cleanly (cans-to-cans >= 0.90, anything else
<= 0.86 to a cans crop).

A change has to hold for CONFIRM checks in a row before the zone's state flips, and then an event is opened on the
camera ("Trash: there"). "Can't see" never flips the state: the cans are still out behind a parked car.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("theeye.watch")

STATES = ("there", "gone", "blocked")
INTERVAL = 60.0          # seconds between checks of each zone
CONFIRM = 3              # same answer this many checks in a row before the state changes
MARGIN = 0.01            # best state must beat the runner-up by this much, else "not sure"
TOP_K = 2                # a state's score: mean of its k most similar examples (1 if it has fewer)


def crop_zone(frame: np.ndarray, points: list[list[float]]) -> Optional[np.ndarray]:
    """The zone's bounding box out of a frame (the whole box: the curb around a can helps tell it apart)."""
    if frame is None or not points:
        return None
    h, w = frame.shape[:2]
    p = np.asarray(points, np.float32) * [w, h]
    x1, y1 = np.floor(p.min(0)).astype(int).clip(0, [w - 1, h - 1])
    x2, y2 = np.ceil(p.max(0)).astype(int).clip(1, [w, h])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2].copy()


def classify(feat: np.ndarray, examples: list[tuple[str, np.ndarray]]) -> tuple[Optional[str], dict[str, float]]:
    """Nearest-examples vote. Returns (state or None if unsure / not taught both there and gone, scores)."""
    by: dict[str, list[float]] = {}
    for st, e in examples:
        by.setdefault(st, []).append(float(e @ feat))
    scores = {st: float(np.mean(sorted(v, reverse=True)[:TOP_K])) for st, v in by.items()}
    if "there" not in scores or "gone" not in scores:
        return None, scores
    ranked = sorted(scores, key=scores.get, reverse=True)
    if len(ranked) > 1 and scores[ranked[0]] - scores[ranked[1]] < MARGIN:
        return None, scores
    return ranked[0], scores


class ZoneWatch:
    def __init__(self, hub):
        self.hub, self.db = hub, hub.db
        self.dir: Path = hub.data / "watch"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._streak: dict[int, tuple[Optional[str], int]] = {}     # zone id -> (raw answer, checks in a row)
        self._last: dict[int, dict] = {}                            # zone id -> last check (answer, scores, time)
        self._wake = threading.Event()

    # ------------------------------------------------------------------ loop
    def start(self) -> None:
        threading.Thread(target=self._loop, name="zone-watch", daemon=True).start()

    def _loop(self) -> None:
        stop = self.hub._stop
        stop.wait(20)                           # let the cameras connect first
        while not stop.is_set():
            for z in self.zones():
                try:
                    self.check(z)
                except Exception:  # noqa: BLE001
                    log.warning("watch zone %s: check failed", z["id"], exc_info=True)
            self._wake.wait(INTERVAL)
            self._wake.clear()

    def zones(self) -> list[dict]:
        rows = self.db.all("SELECT * FROM zones WHERE kind='watch' AND enabled=1 ORDER BY id")
        return [r | {"points": json.loads(r["points"])} for r in rows if r["camera_id"] in self.hub.cams]

    def _look(self, z: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(crop, CLIP picture of it) from the camera's latest main-stream frame; also saved as the zone's 'now' picture."""
        emb = self.hub.embedder
        crop = crop_zone(self.hub._main_frame(z["camera_id"]), z["points"])
        if crop is None or emb is None:
            return crop, None
        cv2.imwrite(str(self.dir / f"zone_{z['id']}_now.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return crop, emb.image(crop)

    def _examples(self, zid: int) -> list[tuple[str, np.ndarray]]:
        return [(r["state"], np.frombuffer(r["embedding"], dtype=np.float32))
                for r in self.db.all("SELECT state, embedding FROM watch_examples WHERE zone_id=?", (zid,))]

    def check(self, z: dict) -> None:
        crop, feat = self._look(z)
        if feat is None:
            return
        answer, scores = classify(feat, self._examples(z["id"]))
        now = time.time()
        with self._lock:
            prev, n = self._streak.get(z["id"], (None, 0))
            n = n + 1 if answer == prev else 1
            self._streak[z["id"]] = (answer, n)
            self._last[z["id"]] = {"answer": answer, "scores": {k: round(v, 3) for k, v in scores.items()}, "ts": now}
        row = self.db.one("SELECT * FROM watch_state WHERE zone_id=?", (z["id"],))
        if row is None:
            self.db.execute("INSERT INTO watch_state(zone_id, state, since, blocked, checked) VALUES(?,?,?,?,?)",
                            (z["id"], None, None, 0, now))
            row = {"state": None, "blocked": 0}
        self.db.execute("UPDATE watch_state SET checked=? WHERE zone_id=?", (now, z["id"]))
        if answer is None or n < CONFIRM:
            return
        if answer == "blocked":
            if not row["blocked"]:
                self.db.execute("UPDATE watch_state SET blocked=1 WHERE zone_id=?", (z["id"],))
            return
        if row["blocked"]:
            self.db.execute("UPDATE watch_state SET blocked=0 WHERE zone_id=?", (z["id"],))
        if answer != row["state"]:
            self._set_state(z, answer, now, announce=row["state"] is not None)

    def _set_state(self, z: dict, state: str, ts: float, announce: bool) -> None:
        self.db.execute("UPDATE watch_state SET state=?, since=?, blocked=0 WHERE zone_id=?", (state, ts, z["id"]))
        what = self.what(z)
        log.info("[%s] watch zone %s: %s %s", z["camera_id"], z["name"], what, state)
        if not announce:
            return                                   # first answer after setup: nothing changed that we saw
        try:
            eid = self.hub.manual_event(z["camera_id"], duration=10, preroll=5, source="watch",
                                        note=f"{what}: {'there' if state == 'there' else 'gone'}")
            self.db.execute("UPDATE events SET labels=? WHERE id=?", (json.dumps([what.lower()]), eid))
        except Exception:  # noqa: BLE001
            log.warning("watch zone %s: couldn't open an event", z["id"], exc_info=True)

    @staticmethod
    def what(z: dict) -> str:
        return (z.get("watch_for") or "").strip() or z["name"]

    # ------------------------------------------------------------------ teaching
    def teach(self, zid: int, state: str) -> dict:
        """'Right now it's <state>': keep the current picture as an example, and believe it straight away."""
        if state not in STATES:
            raise ValueError(state)
        z = next((z for z in self.zones() if z["id"] == zid), None)
        if z is None:
            raise KeyError(zid)
        crop, feat = self._look(z)
        if feat is None:
            raise RuntimeError("no picture from the camera right now")
        now = time.time()
        xid = self.db.insert("INSERT INTO watch_examples(zone_id, state, embedding, path, ts) VALUES(?,?,?,?,?)",
                             (zid, state, feat.tobytes(), None, now))
        rel = f"watch/ex_{xid}.jpg"
        cv2.imwrite(str(self.hub.data / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.execute("UPDATE watch_examples SET path=? WHERE id=?", (rel, xid))
        if not self.db.one("SELECT 1 FROM watch_state WHERE zone_id=?", (zid,)):
            self.db.execute("INSERT INTO watch_state(zone_id, checked) VALUES(?,?)", (zid, now))
        with self._lock:
            self._streak[zid] = (state, CONFIRM)
        if state == "blocked":
            self.db.execute("UPDATE watch_state SET blocked=1 WHERE zone_id=?", (zid,))
        elif self.db.one("SELECT state FROM watch_state WHERE zone_id=?", (zid,))["state"] != state:
            self._set_state(z, state, now, announce=False)   # you saw it yourself: no event for it
        else:
            self.db.execute("UPDATE watch_state SET blocked=0 WHERE zone_id=?", (zid,))
        return {"id": xid}

    def forget(self, xid: int) -> None:
        r = self.db.one("SELECT path FROM watch_examples WHERE id=?", (xid,))
        if not r:
            raise KeyError(xid)
        self.db.execute("DELETE FROM watch_examples WHERE id=?", (xid,))
        if r["path"]:
            (self.hub.data / r["path"]).unlink(missing_ok=True)

    # ------------------------------------------------------------------ reporting
    def status(self, zid: Optional[int] = None) -> list[dict]:
        out = []
        for z in self.zones():
            if zid is not None and z["id"] != zid:
                continue
            st = self.db.one("SELECT * FROM watch_state WHERE zone_id=?", (z["id"],)) or {}
            ex = self.db.all("SELECT id, state, path, ts FROM watch_examples WHERE zone_id=? ORDER BY ts DESC", (z["id"],))
            now_pic = self.dir / f"zone_{z['id']}_now.jpg"
            out.append({
                "zone_id": z["id"], "camera_id": z["camera_id"], "name": z["name"], "what": self.what(z),
                "state": st.get("state"), "since": st.get("since"), "blocked": bool(st.get("blocked")),
                "checked": st.get("checked"), "last": self._last.get(z["id"]),
                "now_picture": f"watch/zone_{z['id']}_now.jpg" if now_pic.exists() else None,
                "examples": ex, "counts": {s: sum(e["state"] == s for e in ex) for s in STATES},
            })
        return out

    def check_now(self) -> None:
        self._wake.set()
