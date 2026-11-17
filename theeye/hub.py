"""Central coordinator: cameras -> motion/detector -> rules -> events -> recorder.

Motion triggers and events are distinct:
  * motion_triggers rows are written for every zone activation, regardless of rules.
  * events are created only when a rule fires (or manually / via API) and own a clip.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

from .camera import Camera
from .config import Config
from .db import DB, jload
from .detector import Detector, Track
from .embeddings import Embedder
from .motion import MotionEngine, Zone
from .recorder import RecordingManager
from .vehicles import VehicleRegistry
from .rules import CamState, Rule, evaluate, is_loud
from . import gpu
from .plates import PlateRefiner
from .delivery import PERSON_LABELS, SERVICES, VEHICLE_LABELS, DeliveryClassifier
from .subtypes import ANIMAL_LABELS, SubtypeClassifier

log = logging.getLogger("theeye.hub")


@dataclass
class OpenEvent:
    id: int
    camera_id: str
    rule_id: Optional[int]
    start: float
    last_active: float
    postroll: float
    max_length: float
    clip_rel: Optional[str]
    labels: set[str] = field(default_factory=set)
    plates: set[str] = field(default_factory=set)
    until: Optional[float] = None        # manual events: fixed end time


class Hub:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data = cfg.data_dir
        self.db = DB(self.data / "theeye.db")
        self.log_path: Optional[Path] = None      # set by __main__ when a log file sits next to the config
        n = max(1, len([c for c in cfg.cameras if c.enabled]))
        per_cam = int(cfg.storage.ram_budget_gb * 1e9 / n)
        self.cams: dict[str, Camera] = {c.id: Camera(c, per_cam) for c in cfg.cameras if c.enabled}
        self.rec = RecordingManager(self.cams, self.data / "clips")
        self.vehicles = VehicleRegistry(self.db, cfg.vehicles)
        self.motion = MotionEngine(self._motion_start, self._motion_end)
        self.embedder = (Embedder(cfg.detector.clip_model, cfg.detector.clip_pretrained, cfg.detector.device)
                         if cfg.detector.embeddings else None)
        self.animals = SubtypeClassifier(cfg.animals, self.embedder) if self.embedder else None
        self.delivery = DeliveryClassifier(self.embedder, cfg.delivery.min_confidence) if self.embedder else None
        self.people = None                         # theeye.people.People, set by __main__ (needs the models folder)
        self.plates = PlateRefiner(self)
        self.detector: Optional[Detector] = None
        if cfg.detector.enabled:
            self.detector = Detector(cfg.detector, self.data / "snaps", self.motion.zones_for,
                                     self._main_frame, self.embedder, self.motion.motion_in_box)
            self.detector.on_track_new = self._track_new
            self.detector.on_track_end = self._track_end
            self.detector.on_frame_tracks = lambda cam, ts, tr: self._evaluate(cam)
        self._lock = threading.RLock()
        self._sound_lock = threading.Lock()
        self._sound: dict[str, deque] = {}                 # cam -> recent (ts, dBFS)
        self._sound_bucket: dict[str, list] = {}           # cam -> [bucket start, peak, power sum, n]
        self._open_triggers: dict[tuple[str, int], int] = {}
        self._open_events: dict[tuple[str, Optional[int]], OpenEvent] = {}
        self._manual_seq = 0
        self._last_detect: dict[str, float] = {}
        self.rules: list[Rule] = []
        self._stop = threading.Event()
        self._recover_after_crash()
        self.reload_zones()
        self.reload_rules()
        self._seed_defaults()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for cam in self.cams.values():
            cam.on_frame(self._on_frame)
            cam.on_audio(self._on_audio)
            cam.start()
        if self.detector:
            self.detector.start()
            threading.Thread(target=self._detector_guard, name="detector-guard", daemon=True).start()
            threading.Thread(target=self._interim_snapshots, name="interim-snapshots", daemon=True).start()
            if self.cfg.detector.plates:
                self.plates.start()
        threading.Thread(target=self._ticker, name="hub-tick", daemon=True).start()
        if not self.db.one("SELECT 1 FROM vehicles LIMIT 1"):     # first run with vehicle tracking: seed from history
            threading.Thread(target=lambda: log.info("vehicles: seeded from %d past sightings", self.vehicles.backfill()),
                             name="vehicle-backfill", daemon=True).start()
        threading.Thread(target=self._retention, name="retention", daemon=True).start()
        threading.Thread(target=self._animal_backfill, name="animal-backfill", daemon=True).start()
        threading.Thread(target=self._delivery_backfill, name="delivery-backfill", daemon=True).start()
        if self.people:
            threading.Thread(target=self.people.scan_past, name="face-scan", daemon=True).start()
        threading.Thread(target=self._repair_interrupted, name="clip-repair", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        now = time.time()
        for k in list(self._open_events):
            self._close_event(k, now)
        with self._lock:
            open_tr = list(self._open_triggers.values())
            self._open_triggers.clear()
        for tid in open_tr:
            self.db.execute("UPDATE motion_triggers SET end_ts=? WHERE id=?", (now, tid))
        for cam in self.cams.values():
            cam.stop()
        if self.detector:
            self.detector.stop()

    # ------------------------------------------------------------ live config
    RESTART_KEYS = {"detector": "AI detection settings", "server": "web server settings",
                    "storage.path": "storage path"}

    def apply_config(self, new: Config) -> list[str]:
        """Apply a new config while running. Cameras/LLM/limits apply live; returns what needs a restart."""
        old = self.cfg
        restart = []
        if new.detector.model_copy(update={"device": old.detector.device}) != old.detector:
            restart.append(self.RESTART_KEYS["detector"])
        if new.detector.device != old.detector.device:      # GPU/CPU switches live
            self.set_ai_device(new.detector.device)
        if new.server != old.server:
            restart.append(self.RESTART_KEYS["server"])
        if new.storage.path != old.storage.path:
            restart.append(self.RESTART_KEYS["storage.path"])

        wanted = {c.id: c for c in new.cameras if c.enabled}
        n = max(1, len(wanted))
        per_cam = int(new.storage.ram_budget_gb * 1e9 / n)
        for cid in list(self.cams):
            if cid not in wanted or _stream_cfg(wanted[cid]) != _stream_cfg(self.cams[cid].cfg):
                self._remove_camera(cid)
            else:
                self.cams[cid].cfg = wanted[cid]           # e.g. sound settings: apply without reconnecting
        for cid, ccfg in wanted.items():
            if cid not in self.cams:
                cam = Camera(ccfg, per_cam)
                cam.on_frame(self._on_frame)
                cam.on_audio(self._on_audio)
                self.cams[cid] = cam
                if not self._stop.is_set():
                    cam.start()
                log.info("[%s] camera (re)started from config", cid)
        for cam in self.cams.values():
            cam.ring.max_bytes = per_cam
        # keep data path unchanged until restart; everything else takes effect now
        new.storage.path = old.storage.path
        self.vehicles.cfg = new.vehicles
        if self.animals:
            self.animals.cfg = new.animals
        if self.delivery:
            self.delivery.min_confidence = new.delivery.min_confidence
        self.cfg = new
        return restart

    def _remove_camera(self, cid: str) -> None:
        now = time.time()
        with self._lock:
            keys = [k for k in self._open_events if k[0] == cid]
            trig = [(k, v) for k, v in self._open_triggers.items() if k[0] == cid]
            for k, _ in trig:
                self._open_triggers.pop(k, None)
        for k in keys:
            self._close_event(k, now)
        for _, tid in trig:
            self.db.execute("UPDATE motion_triggers SET end_ts=? WHERE id=?", (now, tid))
        cam = self.cams.pop(cid, None)
        if cam:
            cam.stop()
        self.motion.forget(cid)
        log.info("[%s] camera stopped", cid)

    # ------------------------------------------------------------ config/db
    def reload_zones(self) -> None:
        rows = self.db.all("SELECT * FROM zones")
        self.motion.set_zones([Zone(r["id"], r["camera_id"], r["name"], jload(r["points"], []), r["kind"],
                                    r["sensitivity"], r["min_area"], r["min_frames"], bool(r["enabled"]),
                                    bool(r.get("ai", 1)))
                               for r in rows])

    def reload_rules(self) -> None:
        self.rules = [Rule.from_row(r) for r in self.db.all("SELECT * FROM rules")]

    def _recover_after_crash(self) -> None:
        """Close triggers/events left open by an unclean shutdown."""
        self.db.execute("UPDATE motion_triggers SET end_ts=start_ts WHERE end_ts IS NULL")
        for e in self.db.all("SELECT id, start_ts, clip_path FROM events WHERE end_ts IS NULL"):
            end = e["start_ts"]
            if e["clip_path"] and (self.data / e["clip_path"]).exists():
                end = max(end, (self.data / e["clip_path"]).stat().st_mtime)
            self.db.execute("UPDATE events SET end_ts=?, note=COALESCE(note,'interrupted') WHERE id=?",
                            (end, e["id"]))

    def _repair_interrupted(self) -> None:
        """Clips cut off by a stop mid-write end in a broken chunk the browser can stall on (blank player)."""
        from .recorder import repair_clip
        for e in self.db.all("SELECT id, clip_path FROM events WHERE note = 'interrupted' AND clip_path IS NOT NULL"):
            p = self.data / e["clip_path"]
            kept = repair_clip(p) if p.exists() else None
            if kept is None:
                self.db.execute("UPDATE events SET clip_path=NULL, note='interrupted, no usable video' WHERE id=?",
                                (e["id"],))
            else:
                self.db.execute("UPDATE events SET note='interrupted, repaired' WHERE id=?", (e["id"],))
            log.info("event %d: interrupted clip %s", e["id"], f"repaired ({kept:.0f} s kept)" if kept else "unusable")

    def _seed_defaults(self) -> None:
        if self.db.one("SELECT 1 FROM rules LIMIT 1"):
            return
        self.db.insert("INSERT INTO rules(name,camera_id,match,conditions,preroll,postroll) VALUES(?,?,?,?,?,?)",
                       ("Person or vehicle", "*", "any",
                        json.dumps([{"type": "object", "labels": ["person"], "zone_id": None, "min_confidence": 0.5},
                                    {"type": "object", "labels": ["car", "truck", "motorcycle", "bus"],
                                     "zone_id": None, "min_confidence": 0.5}]), 10, 10))
        self.reload_rules()

    # ------------------------------------------------------------- pipeline
    def _main_frame(self, cam: str):
        c = self.cams.get(cam)
        return c.decode_latest_frame() if c else None

    def _on_frame(self, cam: str, img, ts: float) -> None:
        camera = self.cams.get(cam)
        if camera is None:
            return
        self.motion.process(cam, img, ts)
        if self.detector and (camera.cfg.always_detect or self.motion.active_zones(cam)
                              or self.detector.active_tracks(cam, within=TRACK_GRACE)):
            c = camera.cfg
            last = self._last_detect.get(cam, 0.0)
            if ts - last >= 1.0 / max(c.detect_fps, 0.1):
                self._last_detect[cam] = ts
                self.detector.submit(cam, img, ts)

    def _motion_start(self, cam: str, zone_id: int, ts: float, area: float) -> None:
        tid = self.db.insert("INSERT INTO motion_triggers(camera_id,zone_id,start_ts,peak_area) VALUES(?,?,?,?)",
                             (cam, zone_id, ts, area))
        with self._lock:
            self._open_triggers[(cam, zone_id)] = tid
            for ev in self._events_for(cam):
                self.db.execute("INSERT INTO event_links VALUES(?,?,?)", (ev.id, "motion", tid))
        self._evaluate(cam)

    def _motion_end(self, cam: str, zone_id: int, ts: float, peak: float) -> None:
        with self._lock:
            tid = self._open_triggers.pop((cam, zone_id), None)
        if tid:
            self.db.execute("UPDATE motion_triggers SET end_ts=?, peak_area=? WHERE id=?", (ts, peak, tid))
        self._evaluate(cam)

    def _track_new(self, t: Track) -> None:
        t.db_id = self.db.insert(
            "INSERT INTO detections(camera_id,track_key,label,confidence,first_ts,last_ts,bbox,zones)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (t.camera_id, t.key, t.label, t.conf, t.first_ts, t.last_ts, json.dumps(t.bbox), "[]"))
        with self._lock:
            for ev in self._events_for(t.camera_id):
                self._link_detection(ev, t)

    def _track_end(self, t: Track) -> None:
        if t.db_id is None:
            return
        if t.hits < 2:      # single-frame blip: discard
            self.db.execute("DELETE FROM detections WHERE id=?", (t.db_id,))
            self.db.execute("DELETE FROM event_links WHERE kind='detection' AND ref_id=?", (t.db_id,))
            return
        emb = self.detector.embed(t) if self.detector else None
        animal = self._classify_animal(t.label, emb) or self._classify_delivery(t.label, emb, t.best_bbox)
        if animal:
            t.kind, t.subtype = animal[0], animal[1]
        snap = self.detector.save_snapshot(t) if self.detector else None    # after: the picture shows the better name
        self.db.execute(
            "UPDATE detections SET confidence=?, last_ts=?, bbox=?, zones=?, plate=?, plate_conf=?,"
            " snapshot_path=COALESCE(?, snapshot_path), embedding=COALESCE(?, embedding), moving=? WHERE id=?",
            (t.best_conf, t.last_ts, json.dumps(t.best_bbox), json.dumps(sorted(t.zones)), t.plate,
             t.plate_conf or None, snap, emb, int(t.travelled), t.db_id))
        vid = None
        if self.vehicles.eligible(t.label, t.best_bbox, t.travelled):
            try:
                vid = self.vehicles.observe(t.db_id, t.camera_id, t.label, t.first_ts, t.last_ts, emb, t.plate,
                                            t.plate_conf, snap)
            except Exception:  # noqa: BLE001
                log.exception("vehicle matching failed")
        if t.plate:
            with self._lock:
                for ev in self._events_for(t.camera_id):
                    ev.plates.add(t.plate)
                    self._flush_event_meta(ev)
        if animal and animal[1] in SERVICES and self.vehicles.is_known(vid):
            animal = None                  # your own / a neighbour's car (marked Known) is never a delivery or police car
        if emb is not None and (t.label in ANIMAL_LABELS or self._delivery_on(t.label)):
            self._store_animal(t.db_id, t.label, animal)
        for row in self.db.all("SELECT event_id FROM event_links WHERE kind='detection' AND ref_id=?", (t.db_id,)):
            self._event_thumb_from_track(row["event_id"], t)
        if self.people and self.cfg.people.enabled and t.label == "person":
            try:
                self.people.on_track_end(t)
            except Exception:  # noqa: BLE001
                log.exception("face recognition failed")

    # -------------------------------------------------------------- animals
    def _classify_animal(self, label: str, emb: Optional[bytes]):
        if not self.animals or emb is None or label not in ANIMAL_LABELS:
            return None
        try:
            vec = np.frombuffer(emb, dtype=np.float32)
            return self._own_pet(label, vec) or self.animals.classify(label, vec)
        except Exception:  # noqa: BLE001
            log.exception("animal classification failed")
            return None

    PET_REFS_TTL = 1800
    PET_MIN_REFS = 10

    def _own_pet(self, label: str, vec: np.ndarray):
        """("dog", <breed>, similarity) if this dog/cat looks like one of your own pets' past sightings."""
        pets = [p.strip().lower() for p in self.cfg.animals.pets if p.strip()]
        if label not in ("cat", "dog") or not pets:
            return None
        now = time.time()
        cache = getattr(self, "_pet_refs", None)
        if cache is None or now - cache[0] > self.PET_REFS_TTL or cache[1] != pets:
            refs = {}
            for p in pets:            # only sightings both models agreed were a dog of this breed, sure of it
                rows = self.db.all("SELECT embedding FROM detections WHERE label='dog' AND kind IS NULL AND subtype=?"
                                   " AND subtype_conf>=0.7 AND embedding IS NOT NULL ORDER BY first_ts DESC LIMIT 300",
                                   (p,))
                if len(rows) >= self.PET_MIN_REFS:
                    m = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                    refs[p] = m / np.linalg.norm(m, axis=1, keepdims=True)
            cache = self._pet_refs = (now, pets, refs)
        v = vec / (np.linalg.norm(vec) or 1.0)
        best = None
        for p, m in cache[2].items():
            s = float(np.sort(m @ v)[-5])           # 5th closest: one look-alike picture isn't enough
            if s >= self.cfg.animals.pet_match and (best is None or s > best[2]):
                best = ("dog", p, s)
        return best

    # ------------------------------------------------------- delivery services
    def _delivery_on(self, label: str) -> bool:
        return bool(self.delivery and self.cfg.delivery.enabled and label in (VEHICLE_LABELS | PERSON_LABELS))

    def _classify_delivery(self, label: str, emb: Optional[bytes], bbox):
        """(label, service, certainty) for an Amazon/UPS/FedEx/USPS vehicle or uniform, in the same shape as an
        animal's closer look, so it's stored, tagged and searched the same way."""
        if emb is None or not self._delivery_on(label):
            return None
        try:
            r = self.delivery.classify(label, np.frombuffer(emb, dtype=np.float32), bbox)
        except Exception:  # noqa: BLE001
            log.exception("delivery classification failed")
            return None
        if r:
            log.info("delivery: %s %s (%.0f%%)", r[0], label, r[1] * 100)
        return (label, r[0], r[1]) if r else None

    def clear_service(self, det_id: int) -> None:
        """Take a delivery/emergency tag off a detection (a family member, or a car marked Known), and off its
        recordings unless something else in them has that tag."""
        r = self.db.one("SELECT subtype FROM detections WHERE id=?", (det_id,))
        if not r or r["subtype"] not in SERVICES:
            return
        tag = r["subtype"]
        self.db.execute("UPDATE detections SET subtype=NULL, subtype_conf=0 WHERE id=?", (det_id,))
        for row in self.db.all("SELECT event_id FROM event_links WHERE kind='detection' AND ref_id=?", (det_id,)):
            eid = row["event_id"]
            if self.db.one("SELECT 1 FROM event_links l JOIN detections d ON d.id=l.ref_id AND l.kind='detection'"
                           " WHERE l.event_id=? AND d.subtype=? LIMIT 1", (eid, tag)):
                continue
            with self._lock:
                ev = next((e for e in self._open_events.values() if e.id == eid), None)
                if ev is not None:
                    ev.labels.discard(tag)
                    self._flush_event_meta(ev)
                    continue
            e = self.db.one("SELECT labels FROM events WHERE id=?", (eid,))
            if e:
                labels = [x for x in jload(e["labels"], []) if x != tag]
                self.db.execute("UPDATE events SET labels=? WHERE id=?", (json.dumps(labels), eid))
        log.info("detection %d: %s tag removed (family member or known vehicle)", det_id, tag)

    def _delivery_backfill(self) -> None:
        """One pass over people and vehicles recorded before this existed."""
        if not self.delivery or not self.cfg.delivery.enabled:
            return
        labels = sorted(VEHICLE_LABELS | PERSON_LABELS)
        n = tagged = 0
        while not self._stop.is_set():
            rows = self.db.all(
                "SELECT id, label, embedding, bbox FROM detections WHERE subtype_conf IS NULL AND embedding IS NOT NULL"
                f" AND label IN ({','.join('?' * len(labels))}) LIMIT 500", labels)
            if not rows:
                break
            for r in rows:
                a = self._classify_delivery(r["label"], r["embedding"], jload(r["bbox"], []))
                self._store_animal(r["id"], r["label"], a)
                n, tagged = n + 1, tagged + bool(a)
        if n:
            log.info("delivery: looked at %d past people/vehicles, %d tagged with a delivery service", n, tagged)

    def _store_animal(self, det_id: int, label: str, animal) -> None:
        """Save the closer look at an animal (subtype_conf 0 = looked, nothing surer than the label), and add
        the better name to the recordings it's in. A corrected kind (squirrel for "bird") also takes the wrong
        label off a recording, unless something else in it really is that."""
        kind, sub, conf = animal if animal else (None, None, 0.0)
        self.db.execute("UPDATE detections SET kind=?, subtype=?, subtype_conf=? WHERE id=?",
                        (kind if kind != label else None, sub, round(conf, 3), det_id))
        if not animal:
            return
        add = {n for n in (kind if kind != label else None, sub) if n}
        for row in self.db.all("SELECT event_id FROM event_links WHERE kind='detection' AND ref_id=?", (det_id,)):
            eid = row["event_id"]
            drop = kind != label and not self.db.one(
                "SELECT 1 FROM event_links l JOIN detections d ON d.id = l.ref_id AND l.kind = 'detection'"
                " WHERE l.event_id = ? AND d.id != ? AND d.label = ? AND d.kind IS NULL LIMIT 1", (eid, det_id, label))
            with self._lock:
                ev = next((e for e in self._open_events.values() if e.id == eid), None)
                if ev is not None:
                    ev.labels |= add
                    if drop:
                        ev.labels.discard(label)
                    self._flush_event_meta(ev)
                    continue
            r = self.db.one("SELECT labels FROM events WHERE id=?", (eid,))
            if r:
                labels = (set(jload(r["labels"], [])) | add) - ({label} if drop else set())
                self.db.execute("UPDATE events SET labels=? WHERE id=?", (json.dumps(sorted(labels)), eid))

    def _animal_backfill(self) -> None:
        """One pass over animal detections recorded before this existed (or while it was off)."""
        if not self.animals or not self.cfg.animals.enabled:
            return
        n = changed = 0
        while not self._stop.is_set():
            rows = self.db.all(
                "SELECT id, label, embedding FROM detections WHERE subtype_conf IS NULL AND embedding IS NOT NULL"
                f" AND label IN ({','.join('?' * len(ANIMAL_LABELS))}) LIMIT 200", sorted(ANIMAL_LABELS))
            if not rows:
                break
            for r in rows:
                a = self._classify_animal(r["label"], r["embedding"])
                self._store_animal(r["id"], r["label"], a)
                n, changed = n + 1, changed + bool(a)
        if n:
            log.info("animals: looked closer at %d past animal detections, %d named more precisely", n, changed)

    # ---------------------------------------------------------------- rules
    def cam_state(self, cam: str) -> CamState:
        tracks = [t.public() for t in self.detector.active_tracks(cam)] if self.detector else []
        with self._sound_lock:
            sound = list(self._sound.get(cam, ()))
        return CamState(self.motion.active_zones(cam), tracks, sound, time.time())

    # ---------------------------------------------------------------- sound
    SOUND_BUCKET_S = 10.0

    def _on_audio(self, cam: str, db: float, ts: float) -> None:
        """~10x/s per camera from its stream thread: keep a minute for rules, log 10 s buckets."""
        with self._sound_lock:
            self._sound.setdefault(cam, deque(maxlen=600)).append((ts, db))
            b = self._sound_bucket.get(cam)
            if b is None or ts - b[0] >= self.SOUND_BUCKET_S:
                if b is not None and b[3]:
                    self.db.execute("INSERT INTO sound_levels(camera_id, ts, peak_db, avg_db) VALUES(?,?,?,?)",
                                    (cam, b[0], round(b[1], 1), round(10 * math.log10(b[2] / b[3] + 1e-12), 1)))
                b = self._sound_bucket[cam] = [ts - ts % self.SOUND_BUCKET_S, -90.0, 0.0, 0]
            b[1] = max(b[1], db)
            b[2] += 10 ** (db / 10)          # average power, not average dB
            b[3] += 1

    def _evaluate(self, cam: str) -> None:
        st = self.cam_state(cam)
        now = time.time()
        nowdt = dt.datetime.now()
        for rule in self.rules:
            if not rule.applies_to(cam) or not rule.in_schedule(nowdt):
                continue
            fired, reasons = evaluate(rule, st)
            if fired:
                self._activate(cam, rule, now, st)
        c = self.cams.get(cam)
        if c and c.cfg.sound_detect and is_loud(st, c.cfg.sound_threshold_db, c.cfg.sound_min_seconds):
            self._activate(cam, self.SOUND_RULE, now, st)

    # Per-camera sound detection (Setup) opens events like a rule would; it has no rules-table row.
    SOUND_RULE = Rule(id=-1, name="Loud sound", preroll=5, postroll=10, max_length=300)

    def _activate(self, cam: str, rule: Rule, now: float, st: CamState) -> None:
        key = (cam, rule.id)
        builtin = rule is self.SOUND_RULE
        with self._lock:
            ev = self._open_events.get(key)
            if ev is None:
                ev = self._open_event(cam, None if builtin else rule.id, "sound" if builtin else "rule", now, rule.preroll, rule.postroll,
                                      rule.max_length, rule.record, rule.name)
                self._open_events[key] = ev
                why = [f"{t['label']} {t['conf']:.2f} hits={t.get('hits')} age={now - t['first_ts']:.0f}s"
                       f"{' drove' if t.get('travelled') else ''}{' still-spot' if t.get('still_spot') else ''}"
                       f" box={[round(v, 2) for v in t['bbox']]}"
                       for t in st.tracks if t.get("confirmed", True) and not t.get("parked")]
                log.info("[%s] event %d triggered by: %s", cam, ev.id, "; ".join(why) or "no object (motion/sound)")
                if rule.webhook:
                    self._webhook(rule.webhook, ev, rule.name)
            ev.last_active = now
            new_labels = {t["label"] for t in st.tracks} - ev.labels
            new_plates = {t["plate"] for t in st.tracks if t.get("plate")} - ev.plates
            if new_labels or new_plates:
                ev.labels |= new_labels
                ev.plates |= new_plates
                self._flush_event_meta(ev)

    def _open_event(self, cam: str, rule_id: Optional[int], source: str, now: float, preroll: float,
                    postroll: float, max_length: float, record: bool, name: str,
                    note: Optional[str] = None) -> OpenEvent:
        start = now - preroll
        clip_rel, offset = (None, 0.0)
        if record and cam in self.cams:
            safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:40]
            clip_rel, offset = self.rec.acquire(cam, start, safe)
        snap = self._save_event_snapshot(cam, now)
        eid = self.db.insert(
            "INSERT INTO events(camera_id,rule_id,source,start_ts,clip_path,clip_offset,snapshot_path,note,trigger_ts)"
            " VALUES(?,?,?,?,?,?,?,?,?)", (cam, rule_id, source, start, clip_rel, offset, snap, note, now))
        ev = OpenEvent(eid, cam, rule_id, start, now, postroll, max_length, clip_rel)
        tid_rows = [tid for (c, _z), tid in self._open_triggers.items() if c == cam]
        for tid in tid_rows:
            self.db.execute("INSERT INTO event_links VALUES(?,?,?)", (eid, "motion", tid))
        if self.detector:
            for t in self.detector.active_tracks(cam):
                self._link_detection(ev, t)
        log.info("[%s] event %d opened (%s)", cam, eid, name)
        return ev

    def _link_detection(self, ev: OpenEvent, t: Track) -> None:
        if t.db_id is None:
            return
        self.db.execute("INSERT INTO event_links VALUES(?,?,?)", (ev.id, "detection", t.db_id))
        if t.label not in ev.labels:                    # tag the recording with everything found in it
            ev.labels.add(t.label)
            self._flush_event_meta(ev)
        self.db.execute("UPDATE detections SET event_id=COALESCE(event_id, ?) WHERE id=?", (ev.id, t.db_id))

    def _flush_event_meta(self, ev: OpenEvent) -> None:
        self.db.execute("UPDATE events SET labels=?, plates=? WHERE id=?",
                        (json.dumps(sorted(ev.labels)), json.dumps(sorted(ev.plates)), ev.id))

    def _events_for(self, cam: str) -> list[OpenEvent]:
        return [e for (c, _r), e in self._open_events.items() if c == cam]

    def _close_event(self, key, now: float) -> None:
        with self._lock:
            ev = self._open_events.pop(key, None)
        if not ev:
            return
        self.db.execute("UPDATE events SET end_ts=? WHERE id=?", (now, ev.id))
        self._flush_event_meta(ev)
        if self.detector:
            for t in self.detector.active_tracks(ev.camera_id):
                if t.db_id and self.db.one("SELECT 1 FROM event_links WHERE event_id=? AND kind='detection'"
                                           " AND ref_id=?", (ev.id, t.db_id)):
                    self._event_thumb_from_track(ev.id, t)
        if ev.clip_rel:
            self.rec.release(ev.camera_id, ev.clip_rel, now)
        log.info("[%s] event %d closed (%.0fs)", ev.camera_id, ev.id, now - ev.start)

    def _ticker(self) -> None:
        while not self._stop.wait(0.5):
            now = time.time()
            for cam in self.cams:
                self._evaluate(cam)
            with self._lock:
                items = list(self._open_events.items())
            for key, ev in items:
                if ev.until is not None:
                    done = now >= ev.until
                else:
                    done = now - ev.last_active > ev.postroll or now - ev.start > ev.max_length
                if done:
                    self._close_event(key, now)

    # ---------------------------------------------------------- manual/API
    def manual_event(self, cam: str, duration: float = 30, preroll: float = 10, note: str = "",
                     source: str = "manual") -> int:
        if cam not in self.cams:
            raise KeyError(cam)
        now = time.time()
        with self._lock:
            self._manual_seq += 1
            key = (cam, f"{source}-{self._manual_seq}")
            ev = self._open_event(cam, None, source, now, preroll, 0, duration + preroll, True, source, note)
            ev.until = now + duration
            self._open_events[key] = ev
        return ev.id

    # -------------------------------------------------------------- helpers
    def _save_event_snapshot(self, cam: str, now: float) -> Optional[str]:
        c = self.cams.get(cam)
        if not c or c.latest_frame is None:
            return None
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        rel = Path("snaps") / cam / day / f"evt_{int(now * 1000)}.jpg"
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), c.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(rel)

    def _event_thumb_from_track(self, eid: int, t: Track) -> None:
        """Replace the event's thumbnail with a crop centred on this object if it's the best view so far.

        The trigger-time snapshot often shows an empty scene (object just entering). The track's best view
        is re-decoded at full resolution from the RAM loop so the crop stays sharp; falls back to the
        analysis-size frame if that moment has already left the loop.
        """
        if not t.best_bbox or t.best_frame is None:
            return
        # Only a view from during this recording, and not of a parked/sitting vehicle: a car parked for hours
        # is linked to every recording on that camera, and its old best view became every one's thumbnail.
        now = time.time()
        if t.parked or (self.detector and self.detector._sitting(t, now)):
            return
        span = self.db.one("SELECT start_ts, end_ts FROM events WHERE id=?", (eid,))
        if not span or not (span["start_ts"] - 1 <= t.best_ts <= (span["end_ts"] or now) + 1):
            return
        x1, y1, x2, y2 = t.best_bbox
        score = t.best_conf * math.sqrt(max(0.0, (x2 - x1) * (y2 - y1)))   # confident and large wins
        row = self.db.one("SELECT thumb_score, snapshot_path FROM events WHERE id=?", (eid,))
        if not row or score <= (row["thumb_score"] or 0):
            return
        cam = self.cams.get(t.camera_id)
        frame = cam.decode_frame_at(t.best_ts) if cam and t.best_ts else None
        img = object_thumb(frame if frame is not None else t.best_frame, t.best_bbox)
        if img is None:
            return
        day = time.strftime("%Y-%m-%d", time.localtime(t.best_ts or time.time()))
        rel = Path("snaps") / t.camera_id / day / f"evt_{eid}_obj_{int(time.time() * 1000)}.jpg"
        p = self.data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.db.execute("UPDATE events SET snapshot_path=?, thumb_score=? WHERE id=?", (str(rel), score, eid))
        old = row["snapshot_path"]
        if old and old != str(rel) and Path(old).name.startswith("evt_"):   # only this event's own thumbnails
            (self.data / old).unlink(missing_ok=True)

    def _webhook(self, url: str, ev: OpenEvent, name: str) -> None:
        payload = {"event_id": ev.id, "camera_id": ev.camera_id, "rule": name, "start_ts": ev.start}

        def send():
            try:
                httpx.post(url, json=payload, timeout=5)
            except Exception as e:  # noqa: BLE001
                log.warning("webhook %s failed: %s", url, e)
        threading.Thread(target=send, daemon=True).start()

    def _interim_snapshots(self) -> None:
        """Snapshots used to be written only when a track ended, so a parked car tracked for hours showed as a
        blank card in Search and Events. Write one for every live track whose best picture improved, at most
        once a minute each."""
        while not self._stop.wait(SNAPSHOT_EVERY_S):
            now = time.time()
            for cam in list(self.cams):
                for t in self.detector.active_tracks(cam, within=TRACK_GRACE):
                    if t.db_id is None or not t.snapshot_dirty or now - t.snapshot_saved < 60:
                        continue
                    try:
                        snap = self.detector.save_snapshot(t)
                        if snap:
                            t.snapshot_saved = now
                            self.db.execute("UPDATE detections SET snapshot_path=? WHERE id=?", (snap, t.db_id))
                    except Exception:  # noqa: BLE001
                        log.exception("[%s] interim snapshot failed", cam)

    def set_ai_device(self, device: str) -> None:
        """Run detection and visual search on "cpu" or the GPU ("cuda:0"), without restarting. On the CPU the
        GPU memory they used is freed (except the CUDA context, ~0.3 GB, which lasts until TheEye restarts)."""
        self.cfg.detector.device = device
        if self.detector:
            self.detector.set_device(device)
        if self.embedder:
            threading.Thread(target=self.embedder.set_device, args=(device,), name="clip-move", daemon=True).start()

    def _detector_guard(self) -> None:
        """If detection hangs or dies, restart the whole process (systemd's Restart=always brings it back).

        Cameras and motion keep running while the detector is hung, so nothing else notices: on 2026-11-15 a
        GPU fault caused by Chrome left YOLO stuck in a CUDA call and no events were recorded for 12 hours.
        A hung CUDA call can't be cancelled from Python, and a fresh process gets a fresh GPU context.
        """
        while not self._stop.wait(15):
            why = self.detector.stalled(DETECT_STALL_S)
            if not why:
                continue
            log.critical("detector %s: restarting TheEye so detection resumes. Thread stacks follow.", why)
            try:
                import faulthandler
                import sys
                for h in logging.getLogger().handlers:
                    h.flush()
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush()
            finally:
                os._exit(3)

    def _retention(self) -> None:
        while not self._stop.wait(300):
            for step in (self._prune_clips, self._prune_snapshots, self._prune_history, self._rotate_log):
                try:
                    step()
                except Exception:  # noqa: BLE001
                    log.exception("retention: %s failed", step.__name__)
            free = shutil.disk_usage(self.data).free
            if free < 5e9:
                log.warning("low disk space: %.1f GB free", free / 1e9)

    def _prune_clips(self) -> None:
        """Delete the oldest clips (all cameras together) until clips fit in storage.max_gb."""
        limit = self.cfg.storage.max_gb * 1e9
        files = sorted((p for p in (self.data / "clips").rglob("*.mp4")), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        active = {w["path"] for w in self.rec.active().values()}
        for p in files:
            if total <= limit:
                break
            rel = str(p.relative_to(self.data))
            if rel in active:
                continue
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            self.db.execute("UPDATE events SET clip_path=NULL WHERE clip_path=?", (rel,))
            log.info("retention: deleted %s", rel)

    def _prune_snapshots(self) -> None:
        """Keep pictures only as far back as video goes (the oldest event that still has its clip).

        Older than that: event thumbnails and detection snapshots are deleted, and so is the detections'
        CLIP search data. Detection rows that carry nothing else (no event, plate or vehicle) are deleted;
        events, plate reads and vehicle sightings stay as text history. A vehicle's own picture is kept.
        """
        row = self.db.one("SELECT MIN(start_ts) AS t FROM events WHERE clip_path IS NOT NULL")
        horizon = row["t"] if row else None
        if not horizon:                       # nothing recorded yet: no reference point
            return
        keep = {r["snapshot_path"] for r in self.db.all("SELECT snapshot_path FROM vehicles WHERE snapshot_path IS NOT NULL")}
        files: list[str] = []
        n_ev = n_strip = n_del = 0
        for _ in range(50):                   # batches, so the database isn't locked for long
            evs = self.db.all("SELECT id, snapshot_path FROM events WHERE clip_path IS NULL AND start_ts < ?"
                              " AND snapshot_path IS NOT NULL LIMIT ?", (horizon, PRUNE_BATCH))
            if evs:
                files += [e["snapshot_path"] for e in evs]
                self.db.execute(f"UPDATE events SET snapshot_path=NULL WHERE id IN ({_ids(evs)})")
                n_ev += len(evs)
            dets = self.db.all(
                "SELECT d.id, d.snapshot_path, (d.plate IS NULL AND d.vehicle_id IS NULL AND e.id IS NULL) AS bare"
                " FROM detections d LEFT JOIN events e ON e.id = d.event_id"
                " WHERE d.last_ts < ? AND (e.id IS NULL OR e.clip_path IS NULL)"
                " AND (d.snapshot_path IS NOT NULL OR d.embedding IS NOT NULL"
                "      OR (d.plate IS NULL AND d.vehicle_id IS NULL AND e.id IS NULL)) LIMIT ?", (horizon, PRUNE_BATCH))
            if dets:
                files += [d["snapshot_path"] for d in dets if d["snapshot_path"]]
                bare = [d for d in dets if d["bare"]]
                rest = [d for d in dets if not d["bare"]]
                if bare:
                    self.db.execute(f"DELETE FROM event_links WHERE kind='detection' AND ref_id IN ({_ids(bare)})")
                    self.db.execute(f"DELETE FROM detections WHERE id IN ({_ids(bare)})")
                if rest:
                    self.db.execute(f"UPDATE detections SET snapshot_path=NULL, embedding=NULL WHERE id IN ({_ids(rest)})")
                n_del, n_strip = n_del + len(bare), n_strip + len(rest)
            if len(evs) < PRUNE_BATCH and len(dets) < PRUNE_BATCH:
                break
        for rel in set(files) - keep:
            (self.data / rel).unlink(missing_ok=True)
        if n_ev or n_strip or n_del:
            log.info("retention: removed pictures older than %s: %d event thumbnails, %d detections trimmed,"
                     " %d detections deleted", time.strftime("%Y-%m-%d %H:%M", time.localtime(horizon)),
                     n_ev, n_strip, n_del)
        for day in (self.data / "snaps").glob("*/*"):    # drop emptied day folders
            if day.is_dir() and not any(day.iterdir()):
                day.rmdir()

    def _prune_history(self) -> None:
        self.db.execute("DELETE FROM motion_triggers WHERE start_ts < ?", (time.time() - MOTION_KEEP_DAYS * 86400,))
        self.db.execute("DELETE FROM event_links WHERE kind='motion' AND ref_id NOT IN (SELECT id FROM motion_triggers)")
        self.db.execute("DELETE FROM sound_levels WHERE ts < ?", (time.time() - SOUND_KEEP_DAYS * 86400,))

    def _rotate_log(self) -> None:
        """Keep the log file (written by run.sh / the service) under LOG_MAX_BYTES, plus one previous file.

        Copy-then-truncate, because the writer holds the file open in append mode and keeps writing to it.
        """
        p = self.log_path
        if not p or not p.is_file() or p.stat().st_size < LOG_MAX_BYTES:
            return
        shutil.copyfile(p, p.with_name(p.name + ".1"))
        with open(p, "r+b") as f:
            f.truncate(0)
        log.info("log rotated; the previous part is in %s.1", p.name)

    def stats(self) -> dict:
        ram = sum(c.ring.stats["bytes"] for c in self.cams.values())
        return {"ram_bytes": ram, "ram_budget": self.cfg.storage.ram_budget_gb * 1e9,
                "recording": self.rec.active(),
                "detector": self.detector.stats if self.detector else None,
                "plates": self.plates.stats,
                "gpu": gpu.stats(),
                "open_events": len(self._open_events),
                "disk_free": shutil.disk_usage(self.data).free}


TRACK_GRACE = 3.0
PRUNE_BATCH = 2000
SNAPSHOT_EVERY_S = 15
DETECT_STALL_S = 90           # one detection batch normally takes well under a second
MOTION_KEEP_DAYS, SOUND_KEEP_DAYS = 30, 7
LOG_MAX_BYTES = 20_000_000


def _ids(rows: list[dict]) -> str:
    return ",".join(str(int(r["id"])) for r in rows)
LIVE_CAM_KEYS = {"sound_detect", "sound_threshold_db", "sound_min_seconds"}   # change without reconnecting


def _stream_cfg(c) -> dict:
    return c.model_dump(exclude=LIVE_CAM_KEYS)


def object_thumb(frame: Optional[np.ndarray], bbox: list[float], out_w: int = 640, out_h: int = 360) -> Optional[np.ndarray]:
    """16:9 crop centred on a normalized bbox: object fills ~40% of the width, shifted inward at frame edges."""
    if frame is None or not bbox:
        return None
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
    aspect = out_w / out_h
    cw = max((x2 - x1) * 2.5, (y2 - y1) * 2.5 * aspect, W * 0.3)   # context around it; never zoom past 30% width
    cw = min(cw, W, H * aspect)
    ch = cw / aspect
    left = min(max((x1 + x2) / 2 - cw / 2, 0), W - cw)
    top = min(max((y1 + y2) / 2 - ch / 2, 0), H - ch)
    crop = frame[int(top):int(top + ch), int(left):int(left + cw)]
    if crop.size == 0:
        return None
    interp = cv2.INTER_AREA if crop.shape[1] > out_w else cv2.INTER_CUBIC
    return cv2.resize(crop, (out_w, out_h), interpolation=interp)
