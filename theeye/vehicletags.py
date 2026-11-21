"""Vehicles tagged by example: "What vehicle is this?" in the player, and later look-alikes get the same tag.

The delivery tags (theeye.delivery) come from text descriptions, which can't tell some vehicles apart: a white
paratransit shuttle with red and blue stripes read as "USPS mail truck" at 99%, while a real mail truck stayed at 89%
and wasn't tagged. A vehicle you tag once becomes an example. A later vehicle takes the tag of its nearest example
when it's close to it and clearly closer to it than to any example with another tag; otherwise the text
descriptions still decide. "Not a delivery" (NONE) examples hold a look-alike back from any tag.

Tested on a week of one street camera: one shuttle example at MATCH found 6 vehicles, all shuttles (4 of them
tagged USPS before); at 0.78 a FedEx van and a mail truck slipped in.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .delivery import SERVICES, VEHICLE_LABELS
from .db import jload

log = logging.getLogger("theeye.vehicletags")

NONE = "none"           # "not a delivery or service vehicle"
MATCH = 0.82            # similarity to an example needed to take its tag
MARGIN = 0.02           # ...and this much closer to it than to an example with any other tag
MIN_SIDE = 0.07         # same as theeye.delivery: smaller boxes don't show what the vehicle is


class VehicleTags:
    def __init__(self, hub):
        self.hub, self.db = hub, hub.db
        self._lock = threading.Lock()
        self._ex: tuple[list[str], Optional[np.ndarray], list[int]] = ([], None, [])
        self.reload()

    def kinds(self) -> list[str]:
        """Tags you can give a vehicle: the services, then your own (delivery.vehicle_kinds, e.g. Short Bus)."""
        own = [k.strip() for k in self.hub.cfg.delivery.vehicle_kinds if k.strip()]
        return list(dict.fromkeys(list(SERVICES) + own))

    def parent(self, tag: Optional[str]) -> Optional[str]:
        """The broader tag this one is a kind of (Short Bus -> Bus: every short bus is a bus), or None."""
        return self.hub.cfg.delivery.vehicle_parents.get(tag) if tag else None

    def with_parents(self, tag: Optional[str]) -> list[str]:
        """The tag and every broader tag it's a kind of: Short Bus -> [Short Bus, Bus]."""
        out = []
        while tag and tag not in out:
            out.append(tag)
            tag = self.parent(tag)
        return out

    def family(self) -> dict[str, list[str]]:
        """Lowercase tag -> every tag that counts as one (bus -> [Bus, Short Bus]), for search."""
        fam: dict[str, list[str]] = {}
        for k in self.kinds():
            for a in self.with_parents(k):
                fam.setdefault(a.lower(), []).append(k)
        return fam

    def reload(self) -> None:
        rows = self.db.all("SELECT detection_id, tag, embedding FROM vehicle_marks WHERE embedding IS NOT NULL")
        with self._lock:
            self._ex = ([r["tag"] for r in rows],
                        np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else None,
                        [r["detection_id"] for r in rows])

    def match(self, feat: Optional[np.ndarray], bbox=None, skip: Optional[int] = None) -> Optional[tuple[str, float]]:
        """(tag, similarity) of the nearest example when it clearly wins, else None (let the text decide)."""
        with self._lock:
            tags, ex, ids = self._ex
        if feat is None or ex is None:
            return None
        if bbox and len(bbox) == 4 and (bbox[2] - bbox[0] < MIN_SIDE or bbox[3] - bbox[1] < MIN_SIDE):
            return None
        s = ex @ feat
        best: dict[str, float] = {}
        for t, v, i in zip(tags, s, ids):
            if i != skip and v > best.get(t, -1.0):
                best[t] = float(v)
        if not best:
            return None
        top = max(best, key=best.get)
        if best[top] < MATCH:
            return None
        # Tags in top's family (Bus, Short Bus) don't compete with it the way unrelated ones (USPS) do: a vehicle
        # that's clearly a bus but too close to call between Bus and Short Bus is still a Bus.
        root = self.with_parents(top)[-1]
        kin = {t for t in best if self.with_parents(t)[-1] == root}
        outside = max((v for t, v in best.items() if t not in kin), default=-1.0)
        if best[top] - outside < MARGIN:
            return None
        # A Bus example means "a full-size bus", so it competes with Short Bus; when they're too close to call,
        # the answer is what both are (Bus).
        rival = max((t for t in kin if t != top), key=best.get, default=None)
        tag = top
        if rival is not None and best[top] - best[rival] < MARGIN:
            tag = next((t for t in self.with_parents(top) if t in self.with_parents(rival)), None)
            if tag is None:
                return None
        return tag, best[top]

    # ------------------------------------------------------------ tagging by hand
    def mark(self, det_id: int, tag: str) -> int:
        """'This is a <tag>' on a vehicle: tag it, keep it as an example, and look again at past vehicles.
        Returns how many past vehicles changed."""
        d = self.db.one("SELECT label, bbox, embedding FROM detections WHERE id=?", (det_id,))
        if not d:
            raise KeyError(det_id)
        if d["label"] not in VEHICLE_LABELS:
            raise ValueError("not a vehicle")
        b = jload(d["bbox"], [])
        if not d["embedding"] or (len(b) == 4 and (b[2] - b[0] < MIN_SIDE or b[3] - b[1] < MIN_SIDE)):
            # a sliver at the edge of the picture (the bottom of a passing truck) looks like any dark car
            raise ValueError("this sighting is too small to learn from: pick a bigger view of the vehicle")
        self.db.execute("INSERT OR REPLACE INTO vehicle_marks(detection_id, tag, embedding, ts) VALUES(?,?,?,?)",
                        (det_id, tag, d["embedding"], time.time()))
        self.reload()
        self.retag(det_id, d["label"], None if tag == NONE else (tag, 1.0))
        return self.rematch_all()

    def forget(self, det_id: int) -> int:
        """Remove a vehicle from the examples, and look at past vehicles again. Returns how many changed."""
        self.db.execute("DELETE FROM vehicle_marks WHERE detection_id=?", (det_id,))
        self.reload()
        return self.rematch_all()

    def retag(self, det_id: int, label: str, tag: Optional[tuple[str, float]]) -> bool:
        """Put this tag (or none) on a detection and its recordings, in place of the one it had. True if changed."""
        cur = self.db.one("SELECT subtype FROM detections WHERE id=?", (det_id,))
        new = tag[0] if tag else None
        if cur is None or cur["subtype"] == new:
            return False
        if cur["subtype"] in self.kinds():
            self.hub.clear_service(det_id)
        if tag:
            self.hub._store_animal(det_id, label, (label, tag[0], tag[1]))
        return True

    def rematch_all(self) -> int:
        """Look at every past vehicle again with the examples as they are now: an example that clearly matches
        decides; otherwise a tagged vehicle gets what the text descriptions say (so a removed example's tags go)."""
        marked = {r["detection_id"] for r in self.db.all("SELECT detection_id FROM vehicle_marks")}
        own = set(self.kinds()) - set(SERVICES)    # tags that only ever come from examples (Short Bus, Bus)
        n, last = 0, -1
        while True:                     # in pages: a week of vehicles' pictures is ~150 MB
            rows = self.db.all("SELECT id, label, bbox, subtype, embedding FROM detections WHERE id > ? AND embedding IS NOT NULL"
                               f" AND label IN ({','.join('?' * len(VEHICLE_LABELS))}) ORDER BY id LIMIT 5000",
                               [last, *sorted(VEHICLE_LABELS)])
            if not rows:
                break
            last = rows[-1]["id"]
            for r in rows:
                if r["id"] in marked:
                    continue
                m = self.match(np.frombuffer(r["embedding"], dtype=np.float32), jload(r["bbox"], []))
                if m is None:
                    if r["subtype"] in own or r["subtype"] in SERVICES:
                        # no example decides it (any more): back to what the text descriptions say, if anything.
                        # Own tags (Short Bus, Bus) only ever come from examples, so they come off.
                        a = self.hub._classify_delivery(r["label"], r["embedding"], jload(r["bbox"], []))
                        n += self.retag(r["id"], r["label"], (a[1], a[2]) if a and a[1] in SERVICES else None)
                    continue
                try:
                    n += self.retag(r["id"], r["label"], None if m[0] == NONE else m)
                except Exception:  # noqa: BLE001
                    log.debug("re-tag of %d failed", r["id"], exc_info=True)
        if n:
            log.info("vehicle tags: %d past vehicles re-tagged after examples", n)
        return n
