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
import shutil
import threading
import time
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
from .rules import CamState, Rule, evaluate

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
        n = max(1, len([c for c in cfg.cameras if c.enabled]))
        per_cam = int(cfg.storage.ram_budget_gb * 1e9 / n)
        self.cams: dict[str, Camera] = {c.id: Camera(c, per_cam) for c in cfg.cameras if c.enabled}
        self.rec = RecordingManager(self.cams, self.data / "clips")
        self.motion = MotionEngine(self._motion_start, self._motion_end)
        self.embedder = (Embedder(cfg.detector.clip_model, cfg.detector.clip_pretrained, cfg.detector.device)
                         if cfg.detector.embeddings else None)
        self.detector: Optional[Detector] = None
        if cfg.detector.enabled:
            self.detector = Detector(cfg.detector, self.data / "snaps", self.motion.zones_for,
                                     self._main_frame, self.embedder, self.motion.motion_in_box)
            self.detector.on_track_new = self._track_new
            self.detector.on_track_end = self._track_end
            self.detector.on_frame_tracks = lambda cam, ts, tr: self._evaluate(cam)
        self._lock = threading.RLock()
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
            cam.start()
        if self.detector:
            self.detector.start()
        threading.Thread(target=self._ticker, name="hub-tick", daemon=True).start()
        threading.Thread(target=self._retention, name="retention", daemon=True).start()

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
    RESTART_KEYS = {"detector": "AI detection settings", "server": "server host/port",
                    "storage.path": "storage path"}

    def apply_config(self, new: Config) -> list[str]:
        """Apply a new config while running. Cameras/LLM/limits apply live; returns what needs a restart."""
        old = self.cfg
        restart = []
        if new.detector != old.detector:
            restart.append(self.RESTART_KEYS["detector"])
        if new.server != old.server:
            restart.append(self.RESTART_KEYS["server"])
        if new.storage.path != old.storage.path:
            restart.append(self.RESTART_KEYS["storage.path"])

        wanted = {c.id: c for c in new.cameras if c.enabled}
        n = max(1, len(wanted))
        per_cam = int(new.storage.ram_budget_gb * 1e9 / n)
        for cid in list(self.cams):
            if cid not in wanted or wanted[cid] != self.cams[cid].cfg:
                self._remove_camera(cid)
        for cid, ccfg in wanted.items():
            if cid not in self.cams:
                cam = Camera(ccfg, per_cam)
                cam.on_frame(self._on_frame)
                self.cams[cid] = cam
                if not self._stop.is_set():
                    cam.start()
                log.info("[%s] camera (re)started from config", cid)
        for cam in self.cams.values():
            cam.ring.max_bytes = per_cam
        # keep data path unchanged until restart; everything else takes effect now
        new.storage.path = old.storage.path
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
        snap = self.detector.save_snapshot(t) if self.detector else None
        emb = self.detector.embed(t) if self.detector else None
        self.db.execute(
            "UPDATE detections SET confidence=?, last_ts=?, bbox=?, zones=?, plate=?, plate_conf=?,"
            " snapshot_path=COALESCE(?, snapshot_path), embedding=COALESCE(?, embedding) WHERE id=?",
            (t.best_conf, t.last_ts, json.dumps(t.best_bbox), json.dumps(sorted(t.zones)), t.plate,
             t.plate_conf or None, snap, emb, t.db_id))
        if t.plate:
            with self._lock:
                for ev in self._events_for(t.camera_id):
                    ev.plates.add(t.plate)
                    self._flush_event_meta(ev)
        for row in self.db.all("SELECT event_id FROM event_links WHERE kind='detection' AND ref_id=?", (t.db_id,)):
            self._event_thumb_from_track(row["event_id"], t)

    # ---------------------------------------------------------------- rules
    def cam_state(self, cam: str) -> CamState:
        tracks = [t.public() for t in self.detector.active_tracks(cam)] if self.detector else []
        return CamState(self.motion.active_zones(cam), tracks)

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

    def _activate(self, cam: str, rule: Rule, now: float, st: CamState) -> None:
        key = (cam, rule.id)
        with self._lock:
            ev = self._open_events.get(key)
            if ev is None:
                ev = self._open_event(cam, rule.id, "rule", now, rule.preroll, rule.postroll,
                                      rule.max_length, rule.record, rule.name)
                self._open_events[key] = ev
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

    def _retention(self) -> None:
        clip_root = self.data / "clips"
        while not self._stop.wait(300):
            try:
                limit = self.cfg.storage.max_gb * 1e9
                files = sorted((p for p in clip_root.rglob("*.mp4")), key=lambda p: p.stat().st_mtime)
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
                cutoff = time.time() - 30 * 86400
                self.db.execute("DELETE FROM motion_triggers WHERE start_ts < ?", (cutoff,))
                free = shutil.disk_usage(self.data).free
                if free < 5e9:
                    log.warning("low disk space: %.1f GB free", free / 1e9)
            except Exception:  # noqa: BLE001
                log.exception("retention pass failed")

    def stats(self) -> dict:
        ram = sum(c.ring.stats["bytes"] for c in self.cams.values())
        return {"ram_bytes": ram, "ram_budget": self.cfg.storage.ram_budget_gb * 1e9,
                "recording": self.rec.active(),
                "detector": self.detector.stats if self.detector else None,
                "open_events": len(self._open_events),
                "disk_free": shutil.disk_usage(self.data).free}


TRACK_GRACE = 3.0


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
