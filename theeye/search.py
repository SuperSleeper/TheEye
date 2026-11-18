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

from . import timeparse
from .config import LLMConfig
from .db import DB, jload
from .delivery import SERVICES
from .subtypes import ANIMAL_LABELS

log = logging.getLogger("theeye.search")

TIME_WORDS = re.compile(
    r"\b(yesterday|today|tonight|last|past|ago|since|before|after|between|recent(ly)?|morning|afternoon|"
    r"evening|night|noon|midnight|hours?|minutes?|days?|weeks?|months?|weekend|mon|tue|wed|thu|fri|sat|sun|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|am|pm|\d{1,2}:\d{2}|\d{1,2}\s?(am|pm))\b")

COCO_HINT = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "bird"]

# Object words -> detector labels, for searches without the LLM. A query that's only these words becomes a
# plain label filter; anything else ("red truck") also keeps a look-alike ranking.
LABEL_WORDS = {
    "person": ["person"], "people": ["person"], "persons": ["person"], "someone": ["person"], "somebody": ["person"],
    "man": ["person"], "men": ["person"], "woman": ["person"], "women": ["person"], "kid": ["person"],
    "kids": ["person"], "child": ["person"], "children": ["person"], "human": ["person"], "humans": ["person"],
    "me": ["person"], "visitor": ["person"], "visitors": ["person"], "anyone": ["person"], "anybody": ["person"],
    "car": ["car"], "cars": ["car"], "truck": ["truck"], "trucks": ["truck"], "pickup": ["truck", "car"],
    "van": ["truck", "car"], "vans": ["truck", "car"], "suv": ["car", "truck"], "suvs": ["car", "truck"],
    "vehicle": ["car", "truck", "bus", "motorcycle"], "vehicles": ["car", "truck", "bus", "motorcycle"],
    "bus": ["bus"], "buses": ["bus"], "motorcycle": ["motorcycle"], "motorcycles": ["motorcycle"],
    "motorbike": ["motorcycle"], "bike": ["bicycle", "motorcycle"], "bikes": ["bicycle", "motorcycle"],
    "bicycle": ["bicycle"], "bicycles": ["bicycle"], "dog": ["dog"], "dogs": ["dog"], "cat": ["cat"],
    "cats": ["cat"], "bird": ["bird"], "birds": ["bird"], "animal": ["dog", "cat", "bird"],
    "animals": ["dog", "cat", "bird"],
}
# Animals as people say them -> the name the closer look (or your naming) stores.
ANIMAL_WORDS = {r"o?possums?": "opossum", r"coons?": "raccoon", r"racoons?": "raccoon", r"bunn(y|ies)": "rabbit",
                r"skunks": "skunk", r"foxes": "fox", r"coyotes": "coyote", r"squirrels": "squirrel"}

# Delivery services as people say them -> the stored tag (lower case for matching).
SERVICE_KEYS = {s.lower() for s in SERVICES} | {"loud exhaust"}
SERVICE_WORDS = {"amazon": "amazon", "prime": "amazon", "ups": "ups", "fedex": "fedex", "fed ex": "fedex",
                 "usps": "usps", "postal": "usps", "post office": "usps", "mailman": "usps", "mail carrier": "usps",
                 "mail truck": "usps", "mail": "usps", "postman": "usps",
                 # emergency services (longer phrases first: "fire truck" before "fire")
                 "police car": "police", "police officer": "police", "police": "police", "cops": "police",
                 "cop": "police", "sheriff": "police", "patrol car": "police",
                 "fire truck": "fire", "fire engine": "fire", "firefighters": "fire", "firefighter": "fire",
                 "fire department": "fire", "fireman": "fire", "fire": "fire",
                 "ambulance": "emt", "paramedics": "emt", "paramedic": "emt", "emts": "emt", "emt": "emt", "ems": "emt",
                 # theeye.exhaust
                 "loud exhausts": "loud exhaust", "loud exhaust": "loud exhaust", "exhausts": "loud exhaust",
                 "exhaust": "loud exhaust", "loud cars": "loud exhaust", "loud car": "loud exhaust",
                 "loud vehicles": "loud exhaust", "loud vehicle": "loud exhaust", "loud trucks": "loud exhaust",
                 "loud truck": "loud exhaust", "loud motorcycles": "loud exhaust", "loud motorcycle": "loud exhaust",
                 "loud engines": "loud exhaust", "loud engine": "loud exhaust", "mufflers": "loud exhaust",
                 "muffler": "loud exhaust", "revving": "loud exhaust", "revs": "loud exhaust"}
DELIVERY_KEYS = {"amazon", "ups", "fedex", "usps"}
EMERGENCY_KEYS = {"police", "fire", "emt"}
STOP = {"a", "an", "the", "any", "some", "of", "in", "on", "at", "by", "near", "with", "from", "to", "and", "or",
        "was", "were", "is", "are", "who", "that", "there", "show", "me", "find", "all", "seen", "came",
        "going", "coming", "walking", "driving", "leaving", "arriving", "out", "my", "our", "few", "couple",
        "event", "events", "recording", "recordings", "clip", "clips"}

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
        "min_confidence": {"type": "number",
                           "description": "minimum detection certainty 0-1 if the user asks for one, e.g. 'over 80%' = 0.8; else 0"},
        "visual": {"type": "string",
                   "description": "visual appearance to match, e.g. 'red pickup truck', 'man in a hoodie'; empty if none"},
        "limit": {"type": "integer"},
        "explanation": {"type": "string", "description": "one short sentence of how the query was interpreted"},
    },
    "required": ["target", "cameras", "labels", "plate", "zones", "start", "end", "hours", "min_confidence",
                 "visual", "limit", "explanation"],
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
- min_confidence: only when the user asks for a certainty/confidence level: "over 80% certainty" / "at least 80% sure" = 0.8, "very sure" / "high confidence" = 0.8, "certain" = 0.9. Otherwise 0.
- visual: colours, clothing, vehicle type/make, carried items, actions — anything a picture would show beyond the label. Empty if none.
- limit: default 50.
- explanation: max 15 words.
Time examples (relative to NOW): "yesterday evening" = yesterday 17:00 to 23:59:59; "last night" = yesterday 20:00 to today 06:00; "this morning" = today 05:00 to 12:00; "past 2 hours" = NOW-2h to NOW.
NOW: {now} ({weekday})
Cameras: {cameras}
Zones: {zones}"""


CERTAINTY = re.compile(
    r"(?:with\s+)?(?:(?:over|above|more than|greater than|at least|min(?:imum)?|>=?)\s*(\d{1,3}(?:\.\d+)?)\s*%"
    r"(?:\s*(?:certain(?:ty)?|confiden(?:ce|t)|sure|match))?"
    r"|(\d{1,3}(?:\.\d+)?)\s*%\s*(?:\+\s*)?(?:certain(?:ty)?|confiden(?:ce|t)|sure)(?:\s+or\s+(?:more|higher|better))?)", re.I)


CERTAINTY_WORDS = re.compile(r"certain|confiden|\bsure\b|%|percent", re.I)


def _min_conf(f: dict) -> float:
    try:
        v = float(f.get("min_confidence") or 0)
    except (TypeError, ValueError):
        return 0.0
    if v > 1:                      # "80" -> 0.8
        v /= 100
    return min(max(v, 0.0), 0.99)


def _apply_certainty(q: str, f: dict) -> dict:
    """'over 80% certainty' etc. -> min_confidence 0.8 (deterministic; small models often miss it)."""
    m = CERTAINTY.search(q)
    if m:
        f["min_confidence"] = float(m.group(1) or m.group(2)) / 100
    elif not CERTAINTY_WORDS.search(q):
        f["min_confidence"] = 0           # small models copy the prompt's example threshold into unrelated queries
    f["min_confidence"] = _min_conf(f)
    return f


def _parse_when(v: str) -> Optional[float]:
    """ISO datetime -> epoch. Repairs a shape small models produce, e.g. '2026-11-13T17:34:51T00:00:00'
    (date + last time given)."""
    try:
        return dt.datetime.fromisoformat(v).timestamp()
    except ValueError:
        d = re.match(r"(\d{4}-\d{2}-\d{2})", v)
        t = re.findall(r"[T ](\d{2}:\d{2}(?::\d{2})?)", v)
        if not d:
            return None
        try:
            return dt.datetime.fromisoformat(f"{d.group(1)}T{t[-1] if t else '00:00:00'}").timestamp()
        except ValueError:
            return None


NO_CLIP_MAX = 12          # sightings with nothing to play, shown after the playable results

class Searcher:
    def __init__(self, cfg: LLMConfig, db: DB, cameras: dict[str, str], embedder=None, animals=None):
        self.cfg = cfg
        self.db = db
        self.cameras = cameras           # id -> name
        self.embedder = embedder
        self.animals = animals           # SubtypeClassifier: the animal / breed / species names it can give
        self.people = None               # theeye.people.People: names of recognised people
        self.pets = None                 # theeye.pets.Pets: names of recognised pets
        self.cars = None                 # theeye.cars.Cars: names of your vehicles

    # --------------------------------------------------------------- LLM
    def interpret(self, q: str) -> dict[str, Any]:
        now = dt.datetime.now()
        zones = [f'{z["name"]} (camera {z["camera_id"]})' for z in self.db.all("SELECT name, camera_id FROM zones")]
        sys = SYSTEM.format(labels=", ".join(COCO_HINT), now=now.isoformat(timespec="seconds"),
                            weekday=now.strftime("%A"),
                            cameras=", ".join(f"{k} = {v}" for k, v in self.cameras.items()) or "none",
                            zones=", ".join(zones) or "none")
        if not self.cfg.enabled:
            return self._fallback(q, "Read without the LLM (it is turned off in Setup)")
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
            return self._fallback(q, f"LLM unavailable ({e.__class__.__name__}); read without it")

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
        tr = timeparse.parse(q)
        if tr.found:                       # plain-code reading of the time beats a small model's
            _set_time(f, tr)
        elif not TIME_WORDS.search(ql):
            f["start"], f["end"], f["hours"] = "", "", []
        return _apply_certainty(q, f)

    def _apply_people(self, q: str, f: dict) -> None:
        """A family member's, pet's or vehicle's name ("sam", "sam yesterday") filters on who was recognised."""
        names = [n for reg in (self.people, self.pets, self.cars) if reg
                 for n in reg.names().values()]
        if not names:
            return
        low = " " + re.sub(r"[^a-z0-9 ]+", " ", q.lower()) + " "
        found = [n for n in names if re.search(rf"\b{re.escape(n.lower())}('?s)?\b", low)]
        if not found:
            return
        f["people"] = found
        f["labels"] = [l for l in (f.get("labels") or [])
                       if l not in ("person", "dog", "cat", "car", "truck", "bus", "motorcycle")]
        vis = (f.get("visual") or "").lower()
        for n in found:
            vis = re.sub(rf"\b{re.escape(n.lower())}('?s)?\b", " ", vis)
        f["visual"] = " ".join(vis.split()) if any(w not in STOP and w not in LABEL_WORDS for w in vis.split()) else ""

    def _apply_animals(self, q: str, f: dict) -> None:
        """Animal, breed and species names in the question ("squirrel", "crow", "pomeranian") become a filter
        on what the closer look at each animal found. A plain "bird" / "dog" / "cat" stays a label filter."""
        low = " " + re.sub(r"[^a-z0-9 ]+", " ", q.lower()) + " "
        found = []
        for word, svc in SERVICE_WORDS.items():      # "mailman" -> usps, "fed ex" -> fedex
            if re.search(rf"\b{word}\b", low):
                found.append(svc)
                low = re.sub(rf"\b{word}\b", " ", low)
        for word, name in ANIMAL_WORDS.items():      # "possum" -> opossum, "coon" -> raccoon
            if re.search(rf"\b{word}\b", low):
                found.append(name)
                low = re.sub(rf"\b{word}\b", " ", low)
        if self.animals and re.search(r"\b(wildlife|wild animals?|critters?|varmints?)\b", low):
            found += sorted(self.animals.other_kinds() - {"squirrel", "bird"})   # the night visitors, not pets/birds
            low = re.sub(r"\b(wildlife|wild animals?|critters?|varmints?)\b", " ", low)
        names = self.animals.names() if self.animals else set()
        for n in sorted(names, key=len, reverse=True):     # "german shepherd" before "shepherd"
            if re.search(rf"\b{re.escape(n)}(e?s)?\b", low):
                found.append(n)
                low = re.sub(rf"\b{re.escape(n)}(e?s)?\b", " ", low)
        if not any(s in DELIVERY_KEYS for s in found) and re.search(r"\b(deliver(y|ies)|delivery (truck|van|driver)s?|courier)\b", low):
            found += sorted(DELIVERY_KEYS)            # "delivery" with no service named: any of them
        if not any(s in EMERGENCY_KEYS for s in found) and re.search(r"\b(emergency|first responders?)\b", low):
            found += sorted(EMERGENCY_KEYS)
        found = list(dict.fromkeys(found))
        if not found:
            return
        services = [s for s in found if s in SERVICE_KEYS]
        f["animals"] = found
        f["labels"] = [l for l in (f.get("labels") or []) if l not in ANIMAL_LABELS]
        if services:                       # "ups truck" / "amazon driver": the service tag is the filter
            f["labels"] = []
        vis = (f.get("visual") or "").lower()
        for n in found:
            vis = re.sub(rf"\b{re.escape(n)}(e?s)?\b", " ", vis)
        for word in SERVICE_WORDS:
            vis = re.sub(rf"\b{word}\b", " ", vis)
        if services:
            vis = re.sub(r"\b(truck|trucks|van|vans|car|cars|driver|drivers|delivery|deliveries|carrier|courier|guy|"
                         r"man|woman|person|people|vehicle|vehicles|engine|officer|officers|emergency)\b", " ", vis)
        f["visual"] = " ".join(vis.split()) if any(w not in STOP for w in vis.split()) else ""

    def _named_cameras(self, text: str) -> tuple[list[str], str]:
        """Cameras whose name appears in the text (e.g. "driveway" -> Driveway), and the text without it."""
        found, low = [], " " + text.lower() + " "
        for cid, name in sorted(self.cameras.items(), key=lambda kv: -len(kv[1] or "")):
            for n in {(name or "").lower(), cid.lower(), cid.lower().replace("_", " ")}:
                if n and re.search(rf"\b{re.escape(n)}\b", low):
                    found.append(cid)
                    low = re.sub(rf"\b{re.escape(n)}\b", " ", low)
                    break
        return found, " ".join(low.split())

    def _fallback(self, q: str, why: str) -> dict:
        """Without the LLM: time words, camera names, object words and certainty are read in plain code;
        whatever is left ("red", "with a dog") ranks results by look."""
        f = {"target": "detections", "cameras": [], "labels": [], "plate": "", "zones": [], "start": "",
             "end": "", "hours": [], "min_confidence": 0, "visual": "", "limit": 50, "explanation": why}
        f = _apply_certainty(q, f)
        text = CERTAINTY.sub(" ", q) if f["min_confidence"] else q
        tr = timeparse.parse(text)
        if tr.found:
            _set_time(f, tr)
        f["cameras"], text = self._named_cameras(tr.rest)
        words = re.findall(r"[a-z0-9']+", text.lower())
        labels = sorted({l for w in words for l in LABEL_WORDS.get(w, [])})
        if any(w in ("event", "events", "recording", "recordings", "clip", "clips") for w in words):
            f["target"] = "events"
        f["labels"] = labels
        extra = [w for w in words if w not in STOP and w not in LABEL_WORDS and not w.isdigit()
                 and w not in ("event", "events", "recording", "recordings", "clip", "clips")]
        f["visual"] = text.strip() if extra else ""
        return f

    # ------------------------------------------------------------ execute
    def search(self, q: str, cameras: Optional[list[str]] = None) -> dict[str, Any]:
        """cameras: the cameras ticked in the UI (None = all). A camera named in the question narrows it further."""
        t0 = time.time()
        f = self.interpret(q)
        self._apply_animals(q, f)
        self._apply_people(q, f)
        target = f.get("target") or "detections"
        if cameras is not None:
            picked = [c for c in cameras if c in self.cameras]
            named = f.get("cameras") or []
            f["cameras"] = [c for c in named if c in picked] if named else picked
            if not f["cameras"]:          # nothing left to search (empty selection, or named one isn't ticked)
                f["explanation"] = ("No cameras are ticked" if not picked else
                                    "The camera named in the search isn't ticked")
                return {"query": q, "filter": f, "target": target, "results": [],
                        "ms": int((time.time() - t0) * 1000)}
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
            ts = _parse_when(v) if v else None
            if ts is not None:
                where.append(f"{col} {op} ?")
                args.append(ts)

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
        where, args = ["(snapshot_path IS NOT NULL OR event_id IS NOT NULL)"], []   # else nothing to show
        self._cam_clause(f, where, args)
        self._time_clauses(f, "first_ts", where, args)
        labels = [l for l in (f.get("labels") or []) if l]
        if labels:
            where.append(f"COALESCE(kind, label) IN ({','.join('?' * len(labels))})")
            args += labels
        who = [n.lower() for n in (f.get("people") or [])]
        if who:
            where.append(f"LOWER(who) IN ({','.join('?' * len(who))})")
            args += who
        names = f.get("animals") or []
        if names:
            q = ",".join("?" * len(names))
            where.append(f"(LOWER(kind) IN ({q}) OR LOWER(subtype) IN ({q}))")
            args += names + names
        plate = (f.get("plate") or "").upper().replace(" ", "")
        if plate:
            where.append("plate GLOB ?")
            args.append(plate if "*" in plate or "?" in plate else f"*{plate}*")
        minc = _min_conf(f)
        if minc:
            where.append("confidence >= ?")
            args.append(minc)
        visual = (f.get("visual") or "").strip()
        limit = max(1, min(int(f.get("limit") or 50), 500))
        pool = 5000 if visual else limit * 4
        # Sightings in a recording first: most parked cars and pets never start one, and 50 of those would
        # crowd out everything you can actually play.
        order = "first_ts DESC" if visual else "(event_id IS NULL), first_ts DESC"
        rows = self.db.all(f"SELECT * FROM detections WHERE {' AND '.join(where)} "
                           f"ORDER BY {order} LIMIT {pool}", args)
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
        # One card per recording: the best-ranked matching object represents it (thumbnail, seek point),
        # and the card lists every kind of object found in that recording with its certainty. Sightings with no
        # recording of their own play from another one on that camera covering the moment, if there is one;
        # those with nothing to play come after the rest (a few, so you still see the object was there).
        out, unplayable, seen_events = [], [], set()
        own_tags = set()
        for r in rows:
            if len(out) >= limit:
                break
            if not r["event_id"] and len(unplayable) < NO_CLIP_MAX:
                cover = self.db.one(
                    "SELECT id FROM events WHERE camera_id=? AND clip_path IS NOT NULL AND start_ts <= ? + 300"
                    " AND start_ts - clip_offset <= ? AND COALESCE(end_ts, start_ts) + 5 >= ?"
                    " ORDER BY start_ts DESC LIMIT 1", (r["camera_id"], r["first_ts"], r["first_ts"], r["first_ts"]))
                if cover:
                    r["event_id"] = cover["id"]
                    own_tags.add(r["id"])
            if not r["event_id"]:
                if len(unplayable) < NO_CLIP_MAX:
                    unplayable.append(r)
                continue
            if r["event_id"] in seen_events:
                continue
            seen_events.add(r["event_id"])
            out.append(r)
        out += unplayable[:max(0, limit - len(out))]
        for r in out:
            r.pop("embedding", None)
            r["bbox"] = jload(r["bbox"], [])
            r["zones"] = jload(r["zones"], [])
            r["animal"] = r.pop("kind", None)       # the column; "kind" below is the result type
            r["kind"] = "detection"
            r["tags"] = (self._event_tags(r["event_id"]) if r["event_id"] and r["id"] not in own_tags
                         else [{"label": r["subtype"] or r["animal"] or r["label"], "conf": r["confidence"]}])
        return out

    def _event_tags(self, eid: int) -> list[dict]:
        return [{"label": t["label"], "conf": t["conf"]} for t in self.db.all(
            "SELECT COALESCE(d.subtype, d.kind, d.label) AS label, MAX(d.confidence) AS conf"
            " FROM event_links l JOIN detections d"
            " ON d.id = l.ref_id AND l.kind = 'detection' WHERE l.event_id = ?"
            " GROUP BY COALESCE(d.subtype, d.kind, d.label) ORDER BY conf DESC",
            (eid,))]

    def _events(self, f: dict) -> list[dict]:
        where, args = ["1=1"], []
        self._cam_clause(f, where, args)
        self._time_clauses(f, "start_ts", where, args)
        for l in f.get("labels") or []:
            where.append("labels LIKE ?")
            args.append(f'%"{l}"%')
        for n in f.get("people") or []:
            where.append("labels LIKE ?")
            args.append(f'%"{n}"%')
        names = f.get("animals") or []
        if names:
            where.append("(" + " OR ".join("labels LIKE ?" for _ in names) + ")")
            args += [f'%"{n}"%' for n in names]
        plate = (f.get("plate") or "").upper().replace("*", "%")
        if plate:
            where.append("plates LIKE ?")
            args.append(f"%{plate}%")
        minc = _min_conf(f)
        if minc:
            labels = [l for l in (f.get("labels") or []) if l]
            where.append("EXISTS (SELECT 1 FROM event_links l JOIN detections d ON d.id = l.ref_id AND l.kind = 'detection'"
                         " WHERE l.event_id = events.id AND d.confidence >= ?"
                         + (f" AND d.label IN ({','.join('?' * len(labels))})" if labels else "") + ")")
            args += [minc] + labels
        limit = max(1, min(int(f.get("limit") or 50), 500))
        rows = self.db.all(f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY start_ts DESC LIMIT ?",
                           args + [limit])
        hours = set(f.get("hours") or [])
        if hours:
            rows = [r for r in rows if dt.datetime.fromtimestamp(r["start_ts"]).hour in hours]
        for r in rows:
            r["kind"] = "event"
            r["labels"], r["plates"] = jload(r["labels"], []), jload(r["plates"], [])
            r["tags"] = self._event_tags(r["id"])
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


def _set_time(f: dict, tr: "timeparse.TimeRange") -> None:
    f["start"] = tr.start.isoformat(timespec="seconds") if tr.start else ""
    f["end"] = tr.end.isoformat(timespec="seconds") if tr.end else ""
    f["hours"] = tr.hours
    f["time_label"] = tr.label
