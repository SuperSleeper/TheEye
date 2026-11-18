"""Read licence plates again from the saved recording, over many frames, while the machine is quiet.

Live reading grabs one full-resolution frame every 1-30 s and keeps the single best-scoring read, so one blurry
frame can win with a wrong character. Here a finished recording with vehicles in it is decoded at full
resolution while a vehicle is in view (about PLATE_FPS frames a second). Each frame is searched for plates in
overlapping tiles, so small plates aren't shrunk away, and every read is kept. Reads of the same plate are
grouped and voted character by character, weighted by the reader's per-character certainty.

California is most common here, so its standard pattern (7ABC123) settles close calls between look-alike
characters (O/0, I/1, B/8...). Reads that don't fit it are taken as out-of-state and left as read.

Runs in one low-priority background thread with its own small plate reader, and pauses when the machine is busy.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("theeye.plates")

PLATE_FPS = 4.0                 # frames read per second of video while a vehicle is in view
MAX_FRAMES = 160                # per recording
MIN_READS = 3                   # a plate needs this many agreeing frames to be trusted
MIN_CONF = 0.6
IDLE_LOAD = 0.5                 # pause while the 1-minute load average is above this share of the CPUs
NICE = 15
VEHICLES = ("car", "truck", "bus", "motorcycle")
LOOKALIKE_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
LOOKALIKE_LETTER = {v: k for k, v in LOOKALIKE_DIGIT.items() if k in "OIZSBG"}
CA_PATTERN = "DLLLDDD"          # California standard passenger plate since 1980, e.g. 7ABC123


@dataclass
class Read:
    text: str
    probs: list[float]
    conf: float
    t: float                    # seconds into the clip
    box: tuple[float, float, float, float]   # normalized x1,y1,x2,y2 in the frame


# ------------------------------------------------------------------------------------------------- voting
def _close(a: str, b: str) -> bool:
    """Same plate allowing OCR slips: up to 2 substitutions, or one character more or less."""
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 2
    if abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        return any(l[:i] + l[i + 1:] == s for i in range(len(l)))
    return False


def group_reads(reads: list[Read]) -> list[list[Read]]:
    """Cluster reads of the same plate (several vehicles can be in one recording)."""
    groups: list[list[Read]] = []
    for r in sorted(reads, key=lambda r: -r.conf):
        for g in groups:
            if _close(g[0].text, r.text):
                g.append(r)
                break
        else:
            groups.append([r])
    return groups


def _ca_fix(chars: list[Counter]) -> Optional[list[str]]:
    """If the plate fits California's pattern apart from look-alike slips, fix those slips."""
    if len(chars) != len(CA_PATTERN):
        return None
    out = []
    for want, votes in zip(CA_PATTERN, chars):
        ranked = [c for c, _ in votes.most_common()]
        ok = [c for c in ranked if (c.isdigit() if want == "D" else c.isalpha())]
        if ok and votes[ok[0]] >= 0.25 * sum(votes.values()):     # a real runner-up of the right kind
            out.append(ok[0])
            continue
        top = ranked[0]
        fixed = LOOKALIKE_DIGIT.get(top) if want == "D" else LOOKALIKE_LETTER.get(top)
        if (top.isdigit() if want == "D" else top.isalpha()):
            out.append(top)
        elif fixed:
            out.append(fixed)
        else:
            return None            # a clear wrong-kind character: not a California plate
    return out


def vote(group: list[Read]) -> tuple[str, float, int]:
    """Best text for one plate from all its reads: (text, certainty 0-1, frames that agree)."""
    lengths = Counter()
    for r in group:
        lengths[len(r.text)] += r.conf
    n = lengths.most_common(1)[0][0]
    same = [r for r in group if len(r.text) == n]
    chars = [Counter() for _ in range(n)]
    for r in same:
        for i, ch in enumerate(r.text):
            p = r.probs[i] if i < len(r.probs) else r.conf
            chars[i][ch] += p
    ca = _ca_fix(chars)
    text = "".join(ca) if ca else "".join(c.most_common(1)[0][0] for c in chars)
    share = [c[text[i]] / max(sum(c.values()), 1e-9) for i, c in enumerate(chars)]
    agree = sum(r.text == text for r in group)
    certainty = float(min(share)) * min(1.0, len(same) / MIN_READS)     # weakest character, few reads count less
    return text, round(certainty, 3), agree


# ------------------------------------------------------------------------------------------------- reading
def _tiles(w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Overlapping half-size tiles (3 x 3), so the plate detector sees plates at a usable size."""
    tw, th = w // 2, h // 2
    xs, ys = [0, w // 4, w - tw], [0, h // 4, h - th]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class PlateRefiner:
    def __init__(self, hub):
        self.hub = hub
        self._alpr = None
        self.stats = {"refined": 0, "queue": None, "last": None}

    def start(self) -> None:
        threading.Thread(target=self._run, name="plate-refiner", daemon=True).start()

    # ------------------------------------------------------------- scheduling
    def _busy(self) -> bool:
        return os.getloadavg()[0] > IDLE_LOAD * (os.cpu_count() or 4)

    def _next(self) -> Optional[dict]:
        db = self.hub.db
        q = (" FROM events e WHERE e.end_ts IS NOT NULL AND e.clip_path IS NOT NULL AND e.plates_refined IS NULL"
             " AND EXISTS (SELECT 1 FROM event_links l JOIN detections d ON d.id = l.ref_id AND l.kind = 'detection'"
             f" WHERE l.event_id = e.id AND d.label IN ({','.join('?' * len(VEHICLES))}))")
        self.stats["queue"] = db.one("SELECT COUNT(*) AS n" + q, VEHICLES)["n"]
        return db.one("SELECT e.*" + q + " ORDER BY e.start_ts DESC LIMIT 1", VEHICLES)

    def _run(self) -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)    # this thread only (Linux)
        except (AttributeError, OSError):
            pass
        stop = self.hub._stop
        while not stop.wait(5):
            d = self.hub.cfg.detector
            if not (d.plates and d.plates_refine):
                continue
            if self._busy():
                stop.wait(60)
                continue
            ev = self._next()
            if ev is None:
                stop.wait(120)
                continue
            if any(w["path"] == ev["clip_path"] for w in self.hub.rec.active().values()):
                stop.wait(30)          # still being written (shouldn't happen once end_ts is set)
                continue
            t0 = time.time()
            try:
                found = self.refine(ev)
            except Exception:  # noqa: BLE001
                log.exception("plate refine failed for event %s", ev["id"])
                found = None
            self.hub.db.execute("UPDATE events SET plates_refined=? WHERE id=?", (time.time(), ev["id"]))
            self.stats["refined"] += 1
            self.stats["last"] = {"event": ev["id"], "plates": found, "seconds": round(time.time() - t0, 1)}
            if found:
                log.info("plates: event %d read again over the recording: %s (%.0f s)", ev["id"],
                         ", ".join(f"{p} {c:.0%} {a}/{n} frames" for p, c, a, n in found), time.time() - t0)

    def _reader(self):
        if self._alpr is None:
            from fast_alpr import ALPR
            import onnxruntime as ort

            def opts():
                so = ort.SessionOptions()
                so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                return so
            dcfg = self.hub.cfg.detector
            cpu = ["CPUExecutionProvider"]
            self._alpr = ALPR(detector_model=dcfg.plate_detector, ocr_model=dcfg.plate_ocr,
                              detector_providers=cpu, ocr_providers=cpu, ocr_device="cpu",
                              detector_sess_options=opts(), ocr_sess_options=opts())
        return self._alpr

    # ------------------------------------------------------------- one event
    def _windows(self, ev: dict) -> tuple[list[tuple[float, float]], list[dict]]:
        dets = self.hub.db.all(
            "SELECT d.id, d.label, d.first_ts, d.last_ts, d.bbox, d.plate, d.plate_conf, d.vehicle_id, d.moving"
            " FROM event_links l JOIN detections d ON d.id = l.ref_id AND l.kind = 'detection'"
            f" WHERE l.event_id = ? AND d.label IN ({','.join('?' * len(VEHICLES))})", (ev["id"], *VEHICLES))
        off = ev["clip_offset"] - ev["start_ts"]
        clip_end = (ev["end_ts"] or ev["start_ts"]) - ev["start_ts"] + ev["clip_offset"]
        dets = [d for d in dets if d["moving"] != 0]      # parked cars: slow to re-read and add stray plates
        spans = []
        for d in dets:
            last = max(d["last_ts"] or d["first_ts"], d["first_ts"])     # a track cut by a restart has no end
            a, b = max(0.0, d["first_ts"] + off - 0.5), min(clip_end, last + off + 0.5)
            if b > a:
                spans.append((a, b))
        spans.sort()
        merged: list[list[float]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [tuple(m) for m in merged], dets

    def read_clip(self, path: Path, windows: list[tuple[float, float]]) -> list[Read]:
        import av
        alpr = self._reader()
        total = sum(b - a for a, b in windows) or 1.0
        step = max(1.0 / PLATE_FPS, total / MAX_FRAMES)
        reads: list[Read] = []
        with av.open(str(path)) as ct:
            vs = ct.streams.video[0]
            vs.thread_type = "AUTO"
            for a, b in windows:
                ct.seek(int(a / vs.time_base), stream=vs, backward=True)
                nxt = a
                for fr in ct.decode(vs):
                    if fr.pts is None:
                        continue
                    t = float(fr.pts * vs.time_base)
                    if t > b:
                        break
                    if t < nxt:
                        continue
                    nxt = t + step
                    reads += self._read_frame(alpr, fr.to_ndarray(format="bgr24"), t)
                    if self.hub._stop.is_set():
                        return reads
        return reads

    def _read_frame(self, alpr, img: np.ndarray, t: float) -> list[Read]:
        h, w = img.shape[:2]
        found: list[Read] = []
        for x1, y1, x2, y2 in _tiles(w, h):
            for r in alpr.predict(img[y1:y2, x1:x2]):
                if not r.ocr or not r.ocr.text:
                    continue
                text = "".join(ch for ch in r.ocr.text.upper() if ch.isalnum())
                if len(text) < 4:
                    continue
                probs = r.ocr.confidence if isinstance(r.ocr.confidence, list) else [float(r.ocr.confidence)] * len(text)
                conf = float(np.mean(probs)) * float(r.detection.confidence)
                bb = r.detection.bounding_box
                box = ((x1 + bb.x1) / w, (y1 + bb.y1) / h, (x1 + bb.x2) / w, (y1 + bb.y2) / h)
                found.append(Read(text, [float(p) for p in probs][:len(text)], conf, t, box))
        keep: list[Read] = []            # the same plate seen in two overlapping tiles: keep the better read
        for r in sorted(found, key=lambda r: -r.conf):
            if all(_iou(r.box, k.box) < 0.3 for k in keep):
                keep.append(r)
        return keep

    def refine(self, ev: dict) -> Optional[list[tuple[str, float, int, int]]]:
        path = self.hub.data / ev["clip_path"]
        if not path.exists():
            return None
        windows, dets = self._windows(ev)
        if not windows:
            return None
        reads = self.read_clip(path, windows)
        results = []
        for g in group_reads(reads):
            text, conf, agree = vote(g)
            if len(g) >= MIN_READS and conf >= MIN_CONF and len(text) >= 5:
                results.append((text, conf, agree, len(g), g))
        if not results:
            return []
        self._save(ev, dets, results)
        return [(t, c, a, n) for t, c, a, n, _ in results]

    def _save(self, ev: dict, dets: list[dict], results: list) -> None:
        """Give each vehicle in the recording the plate seen on it, and tag the recording with all of them."""
        db = self.hub.db
        results = [(self.hub.known_plate(t), c, a, n, g) for t, c, a, n, g in results]   # your vehicles' plates
        off = ev["clip_offset"] - ev["start_ts"]
        for d in dets:
            a, b = d["first_ts"] + off, d["last_ts"] + off
            box = json.loads(d["bbox"] or "[]")
            best = None
            for text, conf, agree, n, g in results:
                during = [r for r in g if a - 0.5 <= r.t <= b + 0.5]
                if not during:
                    continue
                inside = sum(1 for r in during if len(box) == 4 and box[0] <= (r.box[0] + r.box[2]) / 2 <= box[2]
                             and box[1] <= (r.box[1] + r.box[3]) / 2 <= box[3])
                same = bool(d["plate"]) and _close(d["plate"], text)
                score = (same, inside, len(during))
                if best is None or score > best[0]:
                    best = (score, text, conf, agree, n)
            if best is None:
                continue
            _, text, conf, agree, n = best
            if len(results) > 1 and not best[0][0] and not best[0][1]:
                continue               # several plates and no sign which is this vehicle's
            db.execute("UPDATE detections SET plate=?, plate_conf=?, plate_reads=?, plate_agree=? WHERE id=?",
                       (text, conf, n, agree, d["id"]))
            if d["vehicle_id"]:
                self.hub.vehicles.set_plate(d["vehicle_id"], text, conf)
        plates = sorted({t for t, *_ in results})
        with self.hub._lock:
            open_ev = next((e for e in self.hub._open_events.values() if e.id == ev["id"]), None)
            if open_ev is not None:
                open_ev.plates = set(plates)
                self.hub._flush_event_meta(open_ev)
                return
        db.execute("UPDATE events SET plates=? WHERE id=?", (json.dumps(plates), ev["id"]))
