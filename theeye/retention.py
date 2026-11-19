"""Which recordings to delete, and when.

Every clip belongs to one of three kinds, by what its event caught:
  important  people, animals, anything named (your people, pets and vehicles), flagged vehicles, manual recordings
  vehicles   vehicles only: mostly passing traffic
  other      motion or sound with nothing recognised
Each kind can have a maximum age (storage.keep_days; 0 = no limit). When clips run out of room (max useable, or a
camera's share of it), the oldest go first, but a vehicles clip counts as twice its age and an other clip as four
times, so important ones last longest. Events you chose to keep are never deleted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

TIERS = ("important", "vehicles", "other")
WEIGHT = {"important": 1.0, "vehicles": 2.0, "other": 4.0}     # under space pressure, age counts this many times
ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "boat", "train", "airplane"}

# One row per event with a clip: what it caught, and whether anything in it was named or flagged.
EVENTS_SQL = """
SELECT e.id, e.clip_path, e.camera_id, e.start_ts, e.source, e.labels, e.keep,
       EXISTS(SELECT 1 FROM event_links l JOIN detections d ON d.id = l.ref_id
              WHERE l.event_id = e.id AND l.kind = 'detection' AND d.who IS NOT NULL) AS named,
       EXISTS(SELECT 1 FROM event_links l JOIN detections d ON d.id = l.ref_id JOIN vehicles v ON v.id = d.vehicle_id
              WHERE l.event_id = e.id AND l.kind = 'detection' AND v.status = 'flagged') AS flagged
FROM events e WHERE e.clip_path IS NOT NULL
"""


def tier_of(labels: list[str], source: Optional[str] = None, named: bool = False, flagged: bool = False) -> str:
    lab = set(labels or [])
    if named or flagged or source == "manual" or "person" in lab or lab & ANIMALS:
        return "important"
    if lab & VEHICLES:
        return "vehicles"
    return "other"


@dataclass
class Clip:
    rel: str                 # path under the data folder
    camera: str
    size: int
    ts: float                # when it was recorded
    tier: str = "other"      # a clip no event points to (left behind) is the least important
    keep: bool = False


def clips_by_path(rows) -> dict[str, tuple[str, bool, float]]:
    """clip path -> (tier, kept, start) from EVENTS_SQL rows. A clip shared by events takes the most important."""
    out: dict[str, tuple[str, bool, float]] = {}
    for r in rows:
        labels = r["labels"]
        t = tier_of(json.loads(labels) if isinstance(labels, str) else (labels or []), r["source"],
                    bool(r["named"]), bool(r["flagged"]))
        old = out.get(r["clip_path"])
        if old:
            t = min(t, old[0], key=TIERS.index)
        out[r["clip_path"]] = (t, bool(r["keep"]) or bool(old and old[1]), min(r["start_ts"], old[2]) if old else r["start_ts"])
    return out


def plan(clips: list[Clip], now: float, keep_days: dict[str, float], quotas: dict[str, float], max_bytes: float,
         active: set[str] = frozenset()) -> list[tuple[Clip, str]]:
    """The clips to delete, each with why. Pure: the caller deletes them."""
    out: list[tuple[Clip, str]] = []
    left: list[Clip] = []
    for c in clips:
        if c.keep or c.rel in active:
            continue
        days = keep_days.get(c.tier) or 0
        if days > 0 and now - c.ts > days * 86400:
            out.append((c, f"{c.tier} clips kept {days:g} days"))
        else:
            left.append(c)
    gone = {id(c) for c, _ in out}
    used: dict[str, int] = {}
    for c in clips:                  # kept and recording clips still take up room
        if id(c) not in gone:
            used[c.camera] = used.get(c.camera, 0) + c.size
    total = sum(used.values())
    left.sort(key=lambda c: -(now - c.ts) * WEIGHT[c.tier])
    for c in left:
        over_cam = c.camera in quotas and used[c.camera] > quotas[c.camera]
        if not over_cam and total <= max_bytes:
            continue
        out.append((c, "camera over its share" if over_cam else "over max useable"))
        used[c.camera] -= c.size
        total -= c.size
    return out


def usage(clips: list[Clip]) -> dict[str, dict]:
    """Bytes, count and oldest recording per kind (and kept), for the Storage page."""
    out = {t: {"bytes": 0, "clips": 0, "oldest": None} for t in (*TIERS, "kept")}
    for c in clips:
        for k in (c.tier, "kept") if c.keep else (c.tier,):
            u = out[k]
            u["bytes"] += c.size
            u["clips"] += 1
            u["oldest"] = c.ts if u["oldest"] is None else min(u["oldest"], c.ts)
    return out
