"""Animals named by example: "What animal is this?" in the player, and later sightings that look the same.

CLIP's generic descriptions can't tell a skunk or an opossum at night from a blurry dog or bird (a skunk from above
read as "rabbit" 98%), so an animal you name once becomes an example. A later dog/cat/bird sighting is renamed
after its nearest example when it's very close to it, and closer to it than to any of your named pets' confirmed
pictures (your own dog seen at night was the opossum's nearest neighbour, at 0.82). Examples of an ordinary
dog/cat/bird count too: they stop a look-alike from being named after a wild example.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .pets import look_of

log = logging.getLogger("theeye.wildlife")

LABELS = ("dog", "cat", "bird")      # detector labels an animal (wild or not) can come out as
MATCH = 0.85                         # similarity to an example needed to take its name
PET_MARGIN = 0.02                    # ...and this much closer to it than to any named pet's confirmed pictures


class AnimalExamples:
    def __init__(self, hub):
        self.hub, self.db = hub, hub.db
        self._lock = threading.Lock()
        self._ex: tuple[list[str], Optional[np.ndarray], list[int]] = ([], None, [])
        self.reload()

    def kinds(self) -> list[str]:
        ks = [k.strip().lower() for k in self.hub.cfg.animals.kinds if k.strip()]
        return list(dict.fromkeys(ks + list(LABELS)))

    def reload(self) -> None:
        rows = self.db.all("SELECT detection_id, kind, embedding FROM animal_marks WHERE embedding IS NOT NULL")
        with self._lock:
            self._ex = ([r["kind"] for r in rows],
                        np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else None,
                        [r["detection_id"] for r in rows])

    # ------------------------------------------------------------ naming by hand
    def mark(self, det_id: int, kind: str) -> None:
        """'This is a <kind>' on a detection: name it, keep it as an example, and look again at the rest."""
        kind = kind.strip().lower()
        d = self.db.one("SELECT label, who FROM detections WHERE id=?", (det_id,))
        if not d:
            raise KeyError(det_id)
        feat = look_of(self.hub, det_id)
        self.db.execute("INSERT OR REPLACE INTO animal_marks(detection_id, kind, embedding, ts) VALUES(?,?,?,?)",
                        (det_id, kind, feat.tobytes() if feat is not None else None, time.time()))
        self.hub.rename_animal(det_id, kind)
        pets = self.hub.pets
        if kind not in ("dog", "cat") and pets and d["who"] in pets.names().values():
            pets.unmark_detection(det_id)          # a skunk isn't the dog it was taken for
        self.reload()
        threading.Thread(target=self.rematch_all, name="animals-rematch", daemon=True).start()

    # ------------------------------------------------------------ naming by example
    def match(self, det_id: int, feat: Optional[np.ndarray]) -> Optional[tuple[str, float]]:
        with self._lock:
            kinds, ex, ids = self._ex
        if feat is None or ex is None:
            return None
        s = ex @ feat
        for i in np.argsort(-s):
            if ids[i] != det_id:          # not its own example
                break
        else:
            return None
        if s[i] < MATCH:
            return None
        pets = self.hub.pets
        if kinds[i] not in ("dog", "cat") and pets:
            with pets._lock:
                refs = list(pets._refs.values())
            if refs and max(float((r @ feat).max()) for r in refs) + PET_MARGIN > s[i]:
                return None               # looks at least as much like one of your own pets
        return kinds[i], float(s[i])

    def apply(self, det_id: int) -> Optional[str]:
        """Rename one unmarked sighting after its nearest example, if it's close enough. Returns the new name."""
        if self.db.one("SELECT 1 FROM animal_marks WHERE detection_id=?", (det_id,)):
            return None
        d = self.db.one("SELECT label, kind, who FROM detections WHERE id=?", (det_id,))
        if not d or d["label"] not in LABELS:
            return None
        m = self.match(det_id, look_of(self.hub, det_id))
        if not m or m[0] == (d["kind"] or d["label"]):
            return None
        if m[0] in LABELS and d["kind"] in (None, *LABELS):
            return None                   # examples of plain dogs/cats/birds only hold back wild names
        pets = self.hub.pets
        if d["who"] and pets and d["who"] in pets.names().values():
            # match() already checked it's clearly closer to the example than to any pet's confirmed pictures
            if self.db.one("SELECT 1 FROM pet_looks WHERE detection_id=? AND status='confirmed'", (det_id,)):
                return None               # you said it's your pet
            pets.unmark_detection(det_id)
        self.hub.rename_animal(det_id, m[0], conf=m[1])
        return m[0]

    def on_track_end(self, t) -> None:
        if t.label in LABELS and t.db_id is not None and self._ex[1] is not None:
            self.apply(t.db_id)

    def rematch_all(self) -> None:
        n = 0
        for r in self.db.all("SELECT detection_id FROM pet_looks"):      # past dogs/cats already pictured
            try:
                n += bool(self.apply(r["detection_id"]))
            except Exception:  # noqa: BLE001
                log.debug("re-match of %d failed", r["detection_id"], exc_info=True)
        if n:
            log.info("animals: %d past sightings renamed after examples", n)

    def examples(self) -> list[dict]:
        return self.db.all("SELECT m.detection_id, m.kind, d.camera_id, d.first_ts FROM animal_marks m"
                           " JOIN detections d ON d.id = m.detection_id ORDER BY m.ts DESC")
