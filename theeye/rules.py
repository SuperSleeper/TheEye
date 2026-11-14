"""Rule model + evaluation.

A rule is: camera scope + schedule + list of conditions (match any/all) -> create event.

Condition types (JSON):
  {"type": "motion", "zone_id": 3}                  # motion trigger active in zone (null = any zone)
  {"type": "object", "labels": ["person"], "zone_id": null, "min_confidence": 0.5, "min_count": 1}
  {"type": "plate",  "pattern": "ABC*", "zone_id": null}   # glob, case-insensitive; "*" = any plate
Manual / API / webhook events bypass rules.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
from dataclasses import dataclass, field
from typing import Any, Optional

from .db import jload


@dataclass
class Rule:
    id: int
    name: str
    camera_id: str = "*"
    enabled: bool = True
    match: str = "any"
    conditions: list[dict[str, Any]] = field(default_factory=list)
    preroll: float = 10
    postroll: float = 10
    max_length: float = 600
    schedule: Optional[dict] = None
    record: bool = True
    webhook: Optional[str] = None

    @classmethod
    def from_row(cls, r: dict) -> "Rule":
        return cls(id=r["id"], name=r["name"], camera_id=r["camera_id"], enabled=bool(r["enabled"]),
                   match=r["match"], conditions=jload(r["conditions"], []), preroll=r["preroll"],
                   postroll=r["postroll"], max_length=r["max_length"],
                   schedule=jload(r["schedule"]), record=bool(r["record"]), webhook=r["webhook"])

    def applies_to(self, cam: str) -> bool:
        return self.enabled and (self.camera_id in ("*", cam))

    def in_schedule(self, now: dt.datetime) -> bool:
        s = self.schedule
        if not s:
            return True
        days = s.get("days")
        if days is not None and now.weekday() not in days:
            return False
        start, end = s.get("start"), s.get("end")
        if not start or not end:
            return True
        t = now.strftime("%H:%M")
        return (start <= t < end) if start <= end else (t >= start or t < end)   # overnight


@dataclass
class CamState:
    """Snapshot of live signals for one camera, used for evaluation."""
    motion_zones: set[int]               # active motion trigger zone ids
    tracks: list[dict]                   # active tracks (public dicts)


def evaluate(rule: Rule, st: CamState) -> tuple[bool, list[str]]:
    """Returns (fired, reasons)."""
    results: list[tuple[bool, str]] = []
    for c in rule.conditions:
        typ = c.get("type")
        zid = c.get("zone_id")
        if typ == "motion":
            ok = bool(st.motion_zones) if zid in (None, "", "*") else int(zid) in st.motion_zones
            results.append((ok, f"motion{'' if zid in (None, '', '*') else f'@zone{zid}'}"))
        elif typ == "object":
            labels = set(c.get("labels") or [])
            minc = float(c.get("min_confidence", 0))
            matches = [t for t in st.tracks
                       if not t.get("parked")                  # parked cars etc. don't trigger events
                       and (not labels or t["label"] in labels) and t["conf"] >= minc
                       and (zid in (None, "", "*") or int(zid) in t["zones"])]
            ok = len(matches) >= int(c.get("min_count", 1))
            results.append((ok, f"{'/'.join(sorted(labels)) or 'object'} x{len(matches)}"))
        elif typ == "plate":
            pat = (c.get("pattern") or "*").upper()
            pats = [p.strip() for p in pat.split(",") if p.strip()]
            hits = [t["plate"] for t in st.tracks if t.get("plate") and not t.get("parked")
                    and any(fnmatch.fnmatch(t["plate"], p) for p in pats)
                    and (zid in (None, "", "*") or int(zid) in t["zones"])]
            results.append((bool(hits), f"plate {','.join(hits)}" if hits else "plate"))
        else:
            results.append((False, f"unknown:{typ}"))
    if not results:
        return False, []
    fired = all(r[0] for r in results) if rule.match == "all" else any(r[0] for r in results)
    return fired, [r[1] for r in results if r[0]]
