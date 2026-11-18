"""Your vehicles, named: recognised by how they look and by their licence plate, so their arrivals and departures are
tagged with the name. Same flow as pets (theeye.looks).

Only vehicles that drove somewhere are pictured (theeye.vehicles' rule), never parked ones. A confidently read
plate decides on its own: the same plate as one of yours (allowing one OCR slip) tags it, a clearly different one
rules that vehicle out whatever it looks like (another white SUV of the same model). A vehicle's plate is the one
you typed in, else the one read off the first sighting you confirmed; you can correct it.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

from .db import jload
from .looks import NamedLooks
from .vehicles import _close_plates

log = logging.getLogger("theeye.cars")

PLATE_CONF = 0.6                 # plate reads at least this sure count (as for repeat-vehicle tracking)
PLATE_MIN_LEN = 5


def clean_plate(p: Optional[str]) -> Optional[str]:
    p = re.sub(r"[^A-Z0-9]", "", (p or "").upper())
    return p or None


def good_plate(d) -> Optional[str]:
    """The detection's plate when it was read surely enough to go by."""
    p = clean_plate(d and d["plate"])
    return p if p and len(p) >= PLATE_MIN_LEN and (d["plate_conf"] or 0) >= PLATE_CONF else None


class Cars(NamedLooks):
    REG, LOOKS, OWNER, PREFIX = "cars", "car_looks", "car_id", "car"
    LABELS = ("car", "truck", "bus", "motorcycle")
    # Same model and colour look much alike. Measured on 15 vehicles x 8 sightings, 5 confirmed pictures: at 0.88,
    # 47% of its other sightings match and 1.5% of other vehicles do (0.86: 62% / 4.3%). More pictures raise the first.
    MATCH, MARGIN, SUGGEST, MIN_PX, ROOM = 0.88, 0.03, 0.75, 48, 1.05
    SCAN_WHERE = " AND d.vehicle_id IS NOT NULL"       # drove somewhere (theeye.vehicles), not parked

    def __init__(self, hub):
        self._plates: dict[int, str] = {}
        super().__init__(hub)

    def reload(self) -> None:
        super().reload()
        with self._lock:
            self._plates = {r["id"]: r["plate"] for r in self.db.all(f"SELECT id, plate FROM {self.REG}") if r["plate"]}

    def wants(self, t) -> bool:
        return self.hub.vehicles.eligible(t.label, t.best_bbox, t.travelled)

    # ------------------------------------------------------------ plates
    def plate_of(self, det_id: int) -> Optional[str]:
        return good_plate(self.db.one("SELECT plate, plate_conf FROM detections WHERE id=?", (det_id,)))

    def _pick(self, row, feat: np.ndarray) -> tuple[Optional[int], float]:
        plate = self.plate_of(row["detection_id"])
        if plate:
            rejected = set(jload(row["rejected"], []))
            with self._lock:
                for pid, p in self._plates.items():
                    if pid not in rejected and _close_plates(p, plate):
                        return pid, 1.0
        pid, s = super()._pick(row, feat)
        with self._lock:
            mine = self._plates.get(pid) if pid is not None else None
        if plate and mine and not _close_plates(mine, plate):
            return None, s               # looks like it, but the plate says it's another vehicle
        return pid, s

    def set_plate(self, pid: int, plate: Optional[str], source: str = "you") -> Optional[str]:
        """Type in or correct a vehicle's plate (empty clears it). Returns the stored plate."""
        if pid not in self._names:
            raise KeyError(pid)
        plate = clean_plate(plate)
        self.db.execute(f"UPDATE {self.REG} SET plate=?, plate_src=? WHERE id=?",
                        (plate, source if plate else None, pid))
        self.reload()
        self._rematch_soon()
        return plate

    def on_confirm(self, row, pid: int) -> None:
        # Its first confirmed sighting with a sure plate gives it a plate, if you haven't typed one in.
        if not self._plates.get(pid):
            plate = self.plate_of(row["detection_id"])
            if plate:
                self.db.execute(f"UPDATE {self.REG} SET plate=?, plate_src='read' WHERE id=? AND plate IS NULL",
                                (plate, pid))
                self.reload()
        # Yours, so the repeat-vehicle tracker never flags it (and it's never taken for a delivery van).
        d = self.db.one("SELECT vehicle_id FROM detections WHERE id=?", (row["detection_id"],))
        if d and d["vehicle_id"] and not self.hub.vehicles.is_known(d["vehicle_id"]):
            try:
                self.hub.vehicles.set_status(d["vehicle_id"], "known", note=self._names.get(pid))
            except KeyError:
                pass

    def _with_clip(self, looks: list[dict]) -> list[dict]:
        for f in looks:           # the plate read off each sighting, to show next to its picture
            d = self.db.one("SELECT plate, plate_conf, label FROM detections WHERE id=?", (f["detection_id"],))
            f["plate"] = clean_plate(d["plate"]) if d and (d["plate_conf"] or 0) >= PLATE_CONF else None
            f["label"] = d["label"] if d else None
        return super()._with_clip(looks)
