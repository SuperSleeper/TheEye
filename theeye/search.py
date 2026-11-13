"""Natural-language search.

1. Local LLM (Ollama) converts the question into a structured filter (JSON schema-constrained).
2. Filter -> SQL over detections/events.
3. Optional visual description -> CLIP text embedding -> rerank detection snapshots.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from typing import Any, Optional

import httpx
import numpy as np

from .config import LLMConfig
from .db import DB, jload

log = logging.getLogger("theeye.search")

TIME_WORDS = re.compile(
    r"\b(yesterday|today|tonight|last|past|ago|since|before|after|between|recent(ly)?|morning|afternoon|"
    r"evening|night|noon|midnight|hours?|minutes?|days?|weeks?|months?|weekend|mon|tue|wed|thu|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|am|pm|\d{1,2}:\d{2}|\d{1,2}\s?(am|pm))\b")

COCO_HINT = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "bird"]

FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "enum": ["detections", "events", "motion"]},
        "cameras": {"type": "array", "items": {"type": "string"}},
        "labels": {"type": "array", "items": {"type": "string"}},
        "plate": {"type": "string", "description": "plate text or glob, e.g. 'ABC*'; empty if none"},
        "zones": {"type": "array", "items": {"type": "string"}},
        "start": {"type": "string", "description": "ISO local datetime or empty"},
        "end": {"type": "string", "description": "ISO local datetime or empty"},
        "hours": {"type": "array", "items": {"type": "integer"},
                  "description": "restrict to these hours of day (0-23), e.g. night = [22,23,0,1,2,3,4,5]"},
        "visual": {"type": "string",
                   "description": "visual appearance to match, e.g. 'red pickup truck', 'man in a hoodie'; empty if none"},
        "limit": {"type": "integer"},
        "explanation": {"type": "string", "description": "one short sentence of how the query was interpreted"},
    },
    "required": ["target", "cameras", "labels", "plate", "zones", "start", "end", "hours", "visual",
                 "limit", "explanation"],
}

SYSTEM = """You translate a security-camera search question into a JSON filter for an NVR database.
Rules:
- target: "detections" for objects/people/vehicles/plates (default), "events" if the user asks about recorded events/alerts, "motion" for raw motion activity.
- cameras: ONLY if the user explicitly names a camera/location matching the camera list; return its id. Otherwise [] (= all cameras). Never guess.
- labels: only from: {labels}. Map synonyms (man/woman/someone -> person; vehicle -> car, truck, bus, motorcycle; van/pickup -> truck, car; delivery -> person, truck). Empty = any.
- plate: licence plate text if mentioned (uppercase, no spaces). "starting with AB" -> "AB*", "ending in 12" -> "*12", partial -> "*X*". Empty if none.
- zones: zone names from the list if referenced.
- start/end: resolve relative times ("yesterday", "last night", "this morning", "past 2 hours") against NOW. Use ISO format YYYY-MM-DDTHH:MM:SS. Empty if no time constraint.
- hours: only for recurring time-of-day constraints without a specific date (e.g. "at night"). Otherwise [].
- visual: colours, clothing, vehicle type/make, carried items, actions — anything a picture would show beyond the label. Empty if none.
- limit: default 50.
- explanation: max 15 words.
Time examples (relative to NOW): "yesterday evening" = yesterday 17:00 to 23:59:59; "last night" = yesterday 20:00 to today 06:00; "this morning" = today 05:00 to 12:00; "past 2 hours" = NOW-2h to NOW.
NOW: {now} ({weekday})
Cameras: {cameras}
Zones: {zones}"""


class Searcher:
    def __init__(self, cfg: LLMConfig, db: DB, cameras: dict[str, str], embedder=None):
        self.cfg = cfg
        self.db = db
        self.cameras = cameras           # id -> name
        self.embedder = embedder

    # --------------------------------------------------------------- LLM
    def interpret(self, q: str) -> dict[str, Any]:
        now = dt.datetime.now()
        zones = [f'{z["name"]} (camera {z["camera_id"]})' for z in self.db.all("SELECT name, camera_id FROM zones")]
        sys = SYSTEM.format(labels=", ".join(COCO_HINT), now=now.isoformat(timespec="seconds"),
                            weekday=now.strftime("%A"),
                            cameras=", ".join(f"{k} = {v}" for k, v in self.cameras.items()) or "none",
                            zones=", ".join(zones) or "none")
        if not self.cfg.enabled:
            return self._fallback(q, "LLM disabled")
        try:
            r = httpx.post(f"{self.cfg.url}/api/chat", timeout=60, json={
                "model": self.cfg.model, "stream": False, "keep_alive": self.cfg.keep_alive,
                "format": FILTER_SCHEMA, "options": {"temperature": 0},
                "messages": [{"role": "system", "content": sys}, {"role": "user", "content": q}],
            })
            r.raise_for_status()
            f = json.loads(r.json()["message"]["content"])
            f.setdefault("limit", 50)
            return self._ground(q, f)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM interpret failed: %s", e)
            return self._fallback(q, f"LLM unavailable ({e.__class__.__name__}); visual search only")

    def _ground(self, q: str, f: dict) -> dict:
        """Small models tend to invent camera/zone filters. Keep only ones the question mentions."""
        ql = q.lower()

        def mentioned(*names: str) -> bool:
            for n in names:
                n = (n or "").lower()
                if n and (n in ql or any(len(w) >= 4 and w in ql for w in n.replace("_", " ").split())):
                    return True
            return False

        lookup = {k.lower(): k for k in self.cameras} | {v.lower(): k for k, v in self.cameras.items()}
        f["cameras"] = [c for c in (f.get("cameras") or [])
                        if c and c.lower() in lookup and mentioned(lookup[c.lower()], self.cameras[lookup[c.lower()]])]
        f["zones"] = [z for z in (f.get("zones") or []) if mentioned(z)]
        if not TIME_WORDS.search(ql):
            f["start"], f["end"], f["hours"] = "", "", []
        return f

    @staticmethod
    def _fallback(q: str, why: str) -> dict:
        return {"target": "detections", "cameras": [], "labels": [], "plate": "", "zones": [], "start": "",
                "end": "", "hours": [], "visual": q, "limit": 50, "explanation": why}

    # ------------------------------------------------------------ execute
    def search(self, q: str) -> dict[str, Any]:
        t0 = time.time()
        f = self.interpret(q)
        target = f.get("target") or "detections"
        if target == "events":
            results = self._events(f)
        elif target == "motion":
            results = self._motion(f)
        else:
            results = self._detections(f)
        return {"query": q, "filter": f, "target": target, "results": results,
                "ms": int((time.time() - t0) * 1000)}

    def _time_clauses(self, f: dict, col: str, where: list[str], args: list) -> None:
        for key, op in (("start", ">="), ("end", "<=")):
            v = (f.get(key) or "").strip()
            if v:
                try:
                    where.append(f"{col} {op} ?")
                    args.append(dt.datetime.fromisoformat(v).timestamp())
                except ValueError:
                    where.pop()

    def _cam_clause(self, f: dict, where: list[str], args: list, col: str = "camera_id") -> None:
        lookup = {k.lower(): k for k in self.cameras} | {v.lower(): k for k, v in self.cameras.items()}
        cams = sorted({lookup[c.lower()] for c in (f.get("cameras") or []) if c and c.lower() in lookup})
        if cams:
            where.append(f"{col} IN ({','.join('?' * len(cams))})")
            args += cams

    def _zone_ids(self, f: dict) -> list[int]:
        names = [n.lower() for n in (f.get("zones") or [])]
        if not names:
            return []
        return [z["id"] for z in self.db.all("SELECT id, name FROM zones") if z["name"].lower() in names]

    def _detections(self, f: dict) -> list[dict]:
        where, args = ["1=1"], []
        self._cam_clause(f, where, args)
        self._time_clauses(f, "first_ts", where, args)
        labels = [l for l in (f.get("labels") or []) if l]
        if labels:
            where.append(f"label IN ({','.join('?' * len(labels))})")
            args += labels
        plate = (f.get("plate") or "").upper().replace(" ", "")
        if plate:
            where.append("plate GLOB ?")
            args.append(plate if "*" in plate or "?" in plate else f"*{plate}*")
        visual = (f.get("visual") or "").strip()
        limit = max(1, min(int(f.get("limit") or 50), 500))
        pool = 5000 if visual else limit * 4
        rows = self.db.all(f"SELECT * FROM detections WHERE {' AND '.join(where)} "
                           f"ORDER BY first_ts DESC LIMIT {pool}", args)
        hours = set(f.get("hours") or [])
        if hours:
            rows = [r for r in rows if dt.datetime.fromtimestamp(r["first_ts"]).hour in hours]
        zids = set(self._zone_ids(f))
        if zids:
            rows = [r for r in rows if zids & set(jload(r["zones"], []))]
        if visual and self.embedder:
            try:
                tv = self.embedder.text(visual)
                scored = []
                for r in rows:
                    if r["embedding"]:
                        v = np.frombuffer(r["embedding"], dtype=np.float32)
                        if v.shape == tv.shape:
                            scored.append((float(v @ tv), r))
                scored.sort(key=lambda x: -x[0])
                if scored:
                    top = scored[0][0]
                    # keep results reasonably close to the best match; unembedded rows go last
                    unscored = [r for r in rows if not r["embedding"]]
                    rows = [dict(r, score=round(s, 4)) for s, r in scored if s >= top - 0.06] + unscored
            except Exception:  # noqa: BLE001
                log.exception("visual rerank failed")
        out = []
        for r in rows[:limit]:
            r.pop("embedding", None)
            r["bbox"] = jload(r["bbox"], [])
            r["zones"] = jload(r["zones"], [])
            r["kind"] = "detection"
            out.append(r)
        return out

    def _events(self, f: dict) -> list[dict]:
        where, args = ["1=1"], []
        self._cam_clause(f, where, args)
        self._time_clauses(f, "start_ts", where, args)
        for l in f.get("labels") or []:
            where.append("labels LIKE ?")
            args.append(f'%"{l}"%')
        plate = (f.get("plate") or "").upper().replace("*", "%")
        if plate:
            where.append("plates LIKE ?")
            args.append(f"%{plate}%")
        limit = max(1, min(int(f.get("limit") or 50), 500))
        rows = self.db.all(f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY start_ts DESC LIMIT ?",
                           args + [limit])
        hours = set(f.get("hours") or [])
        if hours:
            rows = [r for r in rows if dt.datetime.fromtimestamp(r["start_ts"]).hour in hours]
        for r in rows:
            r["kind"] = "event"
            r["labels"], r["plates"] = jload(r["labels"], []), jload(r["plates"], [])
        return rows

    def _motion(self, f: dict) -> list[dict]:
        where, args = ["1=1"], []
        self._cam_clause(f, where, args)
        self._time_clauses(f, "start_ts", where, args)
        zids = self._zone_ids(f)
        if zids:
            where.append(f"zone_id IN ({','.join('?' * len(zids))})")
            args += zids
        limit = max(1, min(int(f.get("limit") or 50), 500))
        rows = self.db.all(f"SELECT * FROM motion_triggers WHERE {' AND '.join(where)} "
                           f"ORDER BY start_ts DESC LIMIT ?", args + [limit])
        for r in rows:
            r["kind"] = "motion"
        return rows
