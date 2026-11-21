"""A tiny read-only MCP server so a local AI assistant (e.g. OpenJarvis/Jarvis) can ask TheEye what it saw.

TheEye owns all perception; this only lets an assistant *read* detections, events, vehicles and the
natural-language search. It speaks the MCP "Streamable HTTP" JSON-RPC dialect (plain HTTP on loopback,
guarded by the same hook token as /api/hooks). It is deliberately a separate, loopback-only listener:
the main app is HTTPS with a self-signed cert that an MCP client can't verify, and loopback + token is
safe without TLS. Nothing here can change a camera, a rule or a recording.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import threading
import re
import time

import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

log = logging.getLogger("theeye.mcp")

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "theeye", "version": "1.0.0"}


def _ago(ts: float | None) -> str:
    if not ts:
        return "unknown time"
    now = dt.datetime.now()
    when = dt.datetime.fromtimestamp(ts)
    secs = (now - when).total_seconds()
    if secs < 0:
        rel = "just now"
    elif secs < 90:
        rel = f"{int(secs)}s ago"
    elif secs < 5400:
        rel = f"{int(secs / 60)} min ago"
    elif secs < 86400:
        rel = f"{int(secs / 3600)}h ago"
    else:
        rel = f"{int(secs / 86400)}d ago"
    stamp = when.strftime("%a %H:%M") if secs < 86400 else when.strftime("%b %-d %H:%M")
    return f"{stamp} ({rel})"


VEHICLE_LABELS = ("car", "truck", "bus", "motorcycle")


LIVE_MAX = 8                      # concurrent live streams for the dashboard
_live_count = {"n": 0}
_live_lock = threading.Lock()


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


class _Webcam:
    """A local USB webcam the assistant can look through on demand. It is not a TheEye camera: nothing is recorded
    or detected. The device is opened on the first request and released after IDLE_S without one, so the webcam
    stays free for video calls the rest of the time."""

    IDLE_S = 15.0
    WARMUP_FRAMES = 10            # the first frames after opening are dark while auto-exposure settles

    FULL_HOLD_S = 5.0             # after the last full-size use (a viewer, a snapshot, a look) stay at full size this long
    MAX_BAD_PACKETS = 30          # corrupt MJPEG frames in a row before giving up on the stream

    def __init__(self, cid: str, name: str, device: str, size: str, gesture_size: str = "1280x720",
                 gesture_fps: int = 30, gesture_exposure: int = 0) -> None:
        self.id, self.name, self.device, self.size = cid, name, device, size
        self.gesture_size, self.gesture_fps, self.gesture_exposure = gesture_size, gesture_fps, gesture_exposure
        self.frame = None             # latest BGR frame
        self.ts = 0.0
        self.error: str | None = None
        self._jpeg: tuple[float, bytes] | None = None
        self._last_use = 0.0
        self._last_full = 0.0         # when something last needed the full-size picture
        self._open_mode: tuple[str, int] | None = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()

    def _want_mode(self) -> tuple[str, int]:
        if time.time() - self._last_full < self.FULL_HOLD_S:
            return self.size, 15
        return self.gesture_size, self.gesture_fps

    def _set_controls(self, gesture: bool) -> None:
        """Hold the frame rate in dim light (no auto-slowdown to lengthen exposure: that blurs a moving hand). In
        gesture mode also a short manual exposure, so a swiping hand stays sharp; full size goes back to auto."""
        import subprocess
        ctl = ["exposure_dynamic_framerate=0"]
        if gesture and self.gesture_exposure:
            ctl += ["auto_exposure=1", f"exposure_time_absolute={self.gesture_exposure}", "gain=255"]
        else:
            ctl += ["auto_exposure=3"]
        try:
            for c in ctl:                              # one at a time: the C920 rejects exposure_time in auto mode
                subprocess.run(["v4l2-ctl", "-d", self.device, "-c", c], timeout=3, capture_output=True)
        except Exception:  # noqa: BLE001 - v4l2-ctl missing: keep the camera's own settings
            pass

    def _run(self) -> None:
        import av
        size, fps = mode = self._want_mode()
        try:
            c = av.open(self.device, format="v4l2", timeout=(5.0, 5.0),
                        options={"input_format": "mjpeg", "video_size": size, "framerate": str(fps)})
        except Exception as e:  # noqa: BLE001 - busy (a video call) or unplugged
            with self._cond:
                self.error, self._thread = f"the webcam couldn't be opened ({e})", None
                self._cond.notify_all()
            log.info("webcam %s: open failed: %s", self.device, e)
            return
        self._open_mode = mode
        log.info("webcam %s opened at %s/%dfps", self.device, size, fps)
        n = bad = 0
        try:
            stream = c.streams.video[0]
            for pkt in c.demux(stream):
                if pkt.size == 0:
                    continue
                try:
                    frames = pkt.decode()
                except av.error.FFmpegError:
                    bad += 1                              # a corrupt MJPEG frame (USB hiccup): skip it
                    if bad >= self.MAX_BAD_PACKETS:
                        raise
                    continue
                bad = 0
                for frame in frames:
                    n += 1
                    if n == 2:                            # the webcam resets exposure on stream start: set after it
                        self._set_controls(gesture=(mode != (self.size, 15)))
                    if n <= self.WARMUP_FRAMES:
                        continue
                    with self._cond:
                        self.frame, self.ts, self.error = frame.to_ndarray(format="bgr24"), time.time(), None
                        self._cond.notify_all()
                        if time.time() - self._last_use > self.IDLE_S:
                            raise StopIteration
                        if self._want_mode() != mode:
                            log.info("webcam %s: switching to %s", self.device, self._want_mode()[0])
                            raise StopIteration
        except StopIteration:
            pass
        except Exception as e:  # noqa: BLE001
            log.info("webcam %s: stream ended: %s", self.device, e)
        finally:
            c.close()
            with self._cond:
                self._thread, self.frame, self._open_mode = None, None, None
                self._cond.notify_all()
            log.info("webcam %s released", self.device)

    def grab(self, wait: float = 6.0, full: bool = True):
        """The current frame (BGR), opening the webcam if needed. None if it can't be opened (self.error says why).
        full: the caller needs the full-size picture (a viewer, a look); the gesture stream passes False, and while
        only it reads, the webcam runs small and fast. A full-size request reopens it at full size (~1 s)."""
        with self._cond:
            now = time.time()
            self._last_use = now
            if full:
                self._last_full = now
            # wrong size open (a viewer arrived while the gestures had it small): wait for it to reopen
            if self._thread is not None and self._open_mode is not None and self._open_mode != self._want_mode():
                end = now + 3.0
                while self._thread is not None and time.time() < end:
                    self._cond.wait(end - time.time())
            if self._thread is None:
                self.error, self.frame = None, None
                self._thread = threading.Thread(target=self._run, name=f"webcam-{self.id}", daemon=True)
                self._thread.start()
            end = time.time() + wait
            while (self.frame is None or time.time() - self.ts > 2.0) and self.error is None and time.time() < end:
                self._cond.wait(end - time.time())
            return self.frame if self.error is None else None

    LIVE_MIME = 'video/mp4; codecs="avc1.42E01F"'      # H.264 Constrained Baseline, level 3.1 (720p30)
    LIVE_GOP = 10                                      # keyframe (= new MP4 fragment) every ~0.7 s: low delay

    def live(self, write: Callable[[bytes], None]) -> None:
        """Smooth live video for the dashboard: webcam frames encoded to H.264 (CPU, x264 ultrafast) as an endless
        fragmented MP4 -- what TheEye's own cameras give the dashboard's player. Keeps the webcam open while it plays;
        returns when write() fails (the viewer left) or the webcam can't be opened."""
        import io
        from fractions import Fraction
        import av

        chunks: list[bytes] = []

        class _Sink(io.RawIOBase):
            def writable(self):
                return True

            def write(self, b):
                chunks.append(bytes(b))
                return len(b)

        img = self.grab()
        if img is None:
            raise RuntimeError(self.error or "no frame")
        h, w = img.shape[:2]
        out = av.open(_Sink(), mode="w", format="mp4",
                      options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
        st = out.add_stream("libx264", rate=15)
        st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.codec_context.time_base = Fraction(1, 90000)
        st.options = {"preset": "ultrafast", "tune": "zerolatency", "profile": "baseline", "level": "3.1",
                      "g": str(self.LIVE_GOP), "keyint_min": str(self.LIVE_GOP), "crf": "26"}
        t0, last_ts, last_pts = None, 0.0, -1
        try:
            while True:
                with self._cond:
                    self._last_use = self._last_full = time.time()
                    if self._thread is None:              # released (idle elsewhere) or failed: reopen
                        self._cond.release()
                        try:
                            if self.grab() is None:
                                break
                        finally:
                            self._cond.acquire()
                    end = time.time() + 2.0
                    while self.ts == last_ts and self.error is None and time.time() < end:
                        self._cond.wait(end - time.time())
                    if self.error is not None or self.frame is None or self.ts == last_ts:
                        break
                    img, last_ts = self.frame, self.ts
                t0 = last_ts if t0 is None else t0
                fr = av.VideoFrame.from_ndarray(img, format="bgr24")
                fr.pts = max(int((last_ts - t0) * 90000), last_pts + 1)
                fr.time_base = Fraction(1, 90000)
                last_pts = fr.pts
                for pkt in st.encode(fr):
                    out.mux(pkt)
                if chunks:
                    data = b"".join(chunks)
                    chunks.clear()
                    write(data)
        finally:
            try:
                out.close()
            except Exception:  # noqa: BLE001
                pass

    def next_frame(self, after_ts: float, timeout: float = 2.0, full: bool = True):
        """The first frame newer than after_ts as (img, ts), keeping the webcam open; (None, after_ts) if none came."""
        if self.grab(full=full) is None:
            return None, after_ts
        with self._cond:
            self._last_use = time.time()
            end = time.time() + timeout
            while self.ts <= after_ts and self.error is None and self._thread is not None and time.time() < end:
                self._cond.wait(end - time.time())
            if self.frame is None or self.ts <= after_ts:
                return None, after_ts
            return self.frame, self.ts

    def jpeg(self) -> bytes | None:
        import cv2
        img = self.grab()
        if img is None:
            return None
        if self._jpeg is None or self._jpeg[0] != self.ts:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            self._jpeg = (self.ts, buf.tobytes())
        return self._jpeg[1]


class _Vision:
    """Who and what is on a camera *right now*, named with TheEye's own People and Cars matchers.

    People/Cars only name a sighting when its track ends, so someone standing at the door (or a car parked for
    hours) isn't named yet. This asks the same matchers about the live frame instead. Read-only: it calls the
    matchers but never writes faces, looks or tags. Answers are cached per track so a dashboard polling once a
    second costs a face match only every few seconds.
    """

    PERSON_RETRY_S = 3.0        # re-try an unrecognised person (they may turn towards the camera)
    PERSON_KEEP_S = 60.0        # keep a recognised name this long for the same track
    VEHICLE_KEEP_S = 600.0      # parked cars don't change
    SEEN_WITHIN_S = 8.0         # a track counts as "in view" if detected within this long

    def __init__(self, hub: Any) -> None:
        self.hub = hub
        self._cache: dict[tuple[str, str], tuple[float, dict]] = {}
        self._cam_of: dict = {}
        self._lock = threading.Lock()

    def _person(self, track: dict, img) -> dict:
        people = getattr(self.hub, "people", None)
        if people is None or img is None:
            return {"name": None, "why": "face recognition is off"}
        from .faces import MATCH, SUGGEST
        r = people.engine.best_face(img, track["bbox"])
        if r is None:
            return {"name": None, "why": "face not visible"}
        pid, score = people._best_person(r[0], set())
        if pid is not None and score >= MATCH:
            return {"name": people.names().get(pid), "why": f"face match {score:.2f}"}
        if pid is not None and score >= SUGGEST:
            return {"name": None, "maybe": people.names().get(pid), "why": f"face partly matches ({score:.2f})"}
        return {"name": None, "why": "not someone I know"}

    def _vehicle(self, track: dict) -> dict:
        db = self.hub.db
        plate = track.get("plate")
        cars = getattr(self.hub, "cars", None)
        if plate:
            norm = cars.known_plate(plate) if cars else plate
            row = db.one("SELECT name FROM cars WHERE plate = ?", (norm,))
            if row:
                return {"name": row["name"], "why": f"plate {norm}"}
        if track.get("db_id"):
            row = db.one("SELECT who FROM detections WHERE id = ?", (track["db_id"],))
            if row and row.get("who"):
                return {"name": row["who"], "why": "matched earlier"}
            if cars is not None:
                from .looks import tight_look
                look = tight_look(self.hub, track["db_id"])
                feat = look[0] if look is not None else self._live_look(track)
                if feat is not None:
                    scores = sorted(cars._scores(feat, set()).items(), key=lambda kv: -kv[1])
                    if scores:
                        pid, s = scores[0]
                        runner = scores[1][1] if len(scores) > 1 else 0.0
                        if s >= cars.MATCH and s - runner >= cars.MARGIN:
                            return {"name": cars.names().get(pid), "why": f"looks like it ({s:.2f})"}
                        if s >= cars.SUGGEST:
                            return {"name": None, "maybe": cars.names().get(pid),
                                    "why": f"looks similar ({s:.2f}){', plate unreadable' if plate else ''}"}
                        return {"name": None, "why": f"not one of yours (closest {cars.names().get(pid)} {s:.2f})"}
                    return {"name": None, "why": "no named cars to compare with"}
                return {"name": None, "why": "no clear picture of it yet"}
        return {"name": None, "why": "not one of yours"}

    def _live_look(self, track: dict):
        """CLIP picture of a vehicle cropped from the live frame. A sighting only gets its snapshot when its track
        ends, so a car that is still parked has none yet; crop it the way looks.tight_look crops a snapshot."""
        from .looks import INSET
        emb = getattr(self.hub, "embedder", None)
        cam = self.hub.cams.get(track.get("camera_id") or self._cam_of.get(track.get("key"), ""))
        if emb is None or cam is None:
            return None
        img = cam.latest_frame            # the detector's frame: the same one confirmed looks were cropped from
        if img is None:
            img = cam.decode_latest_frame()
        b = track.get("bbox") or []
        if img is None or len(b) != 4:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        if max(x2 - x1, y2 - y1) < 48:
            return None
        dx, dy = (x2 - x1) * INSET, (y2 - y1) * INSET
        crop = img[int(y1 + dy):int(y2 - dy), int(x1 + dx):int(x2 - dx)]
        return emb.image(crop) if crop.size else None

    def identify(self, img, min_px: int = 32) -> list[dict]:
        """Every face in a picture, named with TheEye's People matcher (read-only). For pictures from elsewhere,
        e.g. Vector the robot's camera: [{name, maybe, score, size, box}] biggest face first."""
        people = getattr(self.hub, "people", None)
        if people is None or img is None:
            return []
        from .faces import MATCH, SUGGEST
        eng = people.engine
        out = []
        with eng._lock:
            if not eng._load():
                return []
            h, w = img.shape[:2]
            eng._det.setInputSize((w, h))
            _, found = eng._det.detect(img)
            feats = []
            for f in (found if found is not None else []):
                size = int(min(f[2], f[3]))
                if size < min_px:
                    continue
                aligned = eng._rec.alignCrop(img, f)
                feat = eng._rec.feature(aligned).flatten().astype(np.float32)
                feats.append((feat / (np.linalg.norm(feat) or 1.0), size, [round(float(v), 1) for v in f[:4]]))
        from .faces import similarity
        names = people.names()
        with people._lock:
            refs = dict(people._refs)
        for feat, size, box in sorted(feats, key=lambda x: -x[1]):
            ranked = sorted(((similarity(feat, r), pid) for pid, r in refs.items()), reverse=True)
            score, pid = ranked[0] if ranked else (0.0, None)
            lead = score - (ranked[1][0] if len(ranked) > 1 else 0.0)
            name = names.get(pid) if pid is not None else None
            # A "maybe" must also stand out from the next person: small, backlit faces from below score everyone
            # about the same (seen from Vector's camera: 0.31 / 0.30 / 0.30), and that is noise, not a guess.
            out.append({"name": name if score >= MATCH and lead >= 0.03 else None,
                        "maybe": name if SUGGEST <= score < MATCH and lead >= 0.05 else None,
                        "score": round(float(score), 2), "lead": round(float(lead), 2), "size": size, "box": box})
        return out

    def look(self, cam_id: str) -> dict:
        """{camera, ts, objects: [{key, label, bbox, parked, name, why}], recent: [...]}"""
        hub = self.hub
        cam = hub.cams.get(cam_id)
        # Tolerate a few missed detections: "active" in TheEye means seen in the last 1.5 s, and a parked car (or
        # someone standing still) can drop out for a frame or two. 8 s is still "right now".
        det = getattr(hub, "detector", None)
        if cam is None:
            tracks = []
        elif det is not None:
            tracks = [t.public() for t in det.active_tracks(cam_id, within=self.SEEN_WITHIN_S)]
        else:
            tracks = hub.cam_state(cam_id).tracks
        now = time.time()
        img = None
        objects = []
        for t in tracks:
            label = t.get("label")
            if label != "person" and label not in VEHICLE_LABELS:
                continue
            key = (cam_id, str(t.get("key")))
            with self._lock:
                hit = self._cache.get(key)
            fresh = None
            if hit:
                age, res = now - hit[0], hit[1]
                if label == "person":
                    fresh = res if (res["name"] and age < self.PERSON_KEEP_S) or age < self.PERSON_RETRY_S else None
                else:
                    fresh = res if age < self.VEHICLE_KEEP_S else None
            if fresh is None:
                try:
                    if label == "person":
                        if img is None:
                            img = cam.decode_latest_frame()          # full resolution: faces need the pixels
                            if img is None:
                                img = cam.latest_frame
                        fresh = self._person(t, img)
                    else:
                        fresh = self._vehicle({**t, "camera_id": cam_id})
                except Exception as e:  # noqa: BLE001
                    log.debug("look %s %s failed: %s", cam_id, label, e)
                    fresh = {"name": None, "why": "couldn't check"}
                with self._lock:
                    self._cache[key] = (now, fresh)
            objects.append({"key": t.get("key"), "label": label, "bbox": t.get("bbox"),
                            "parked": bool(t.get("parked")), "name": fresh["name"], "maybe": fresh.get("maybe"),
                            "why": fresh["why"]})
        with self._lock:                       # forget tracks not seen for a while
            for k in [k for k, v in self._cache.items() if now - v[0] > self.VEHICLE_KEEP_S * 2]:
                del self._cache[k]
        recent = self.hub.db.all(
            "SELECT label, who, MAX(last_ts) AS ts FROM detections WHERE camera_id = ? AND who IS NOT NULL "
            "AND last_ts > ? GROUP BY label, who ORDER BY ts DESC LIMIT 8", (cam_id, now - 900))
        return {"camera": cam_id, "ts": now, "objects": objects, "recent": recent}


class _Tools:
    """The read-only tool surface. Each method returns a plain-text string for the assistant to read."""

    def __init__(self, hub: Any, searcher: Any, db: Any) -> None:
        self.hub = hub
        self.searcher = searcher
        self.db = db
        self.vision = _Vision(hub)
        srv = getattr(getattr(hub, "cfg", None), "server", None)
        self.webcams = {w.id: _Webcam(w.id, w.name, w.device, w.size, w.gesture_size, w.gesture_fps, w.gesture_exposure) for w in (getattr(srv, "mcp_webcams", None) or [])}
        self.last_shown: dict = {"ts": 0.0, "ids": [], "title": ""}
        self.last_query: dict = {"ts": 0.0}

    # id <-> display name, accepting either from the model
    def _cam_id(self, cam: str | None) -> str | None:
        if not cam:
            return None
        names = self._all_names()  # {id: name}
        if cam in names:
            return cam
        for cid, name in names.items():
            if name.lower() == cam.lower():
                return cid
        key = _canon(re.sub(r"\b(the|camera|cam)\b", " ", cam.lower()))     # "the backyard camera" -> backyard
        for cid, name in names.items():
            if key and key == _canon(name):
                return cid
        if key in ("frontdoor", "door", "porch"):
            for cid, name in names.items():
                if _canon(name) == "frontporch":
                    return cid
        # Speech-to-text spellings of camera names: "drive way", "front yaard" -> Driveway, Front Yard.
        import difflib
        by_key = {_canon(n): cid for cid, n in names.items()}
        close = difflib.get_close_matches(key, list(by_key), n=1, cutoff=0.72)
        if close:
            return by_key[close[0]]
        return cam  # unknown: let the query return nothing rather than guess

    def _all_names(self) -> dict:
        return {**self.searcher.cameras, **{w.id: w.name for w in self.webcams.values()}}

    def _cam_name(self, cid: str | None) -> str:
        return self._all_names().get(cid, cid or "?")

    def webcam_look(self, cid: str) -> dict:
        """look() for a webcam: the faces in its current frame, named by People. Same shape as _Vision.look."""
        w = self.webcams[cid]
        img = w.grab()
        objects = []
        if img is not None:
            h, wd = img.shape[:2]
            for f in self.vision.identify(img, min_px=40):
                x, y, bw, bh = f["box"]
                why = f"face match {f['score']:.2f}" if f["name"] or f["maybe"] else "face doesn't match anyone I know"
                objects.append({"key": None, "label": "person", "bbox": [x / wd, y / h, (x + bw) / wd, (y + bh) / h],
                                "parked": False, "name": f["name"], "maybe": f["maybe"], "why": why})
        return {"camera": cid, "ts": w.ts or time.time(), "objects": objects, "recent": [], "error": w.error}

    # ---- tool implementations ------------------------------------------------
    def list_cameras(self) -> str:
        names = self.searcher.cameras
        if not names:
            return "No cameras are configured."
        out = "Cameras TheEye watches:\n" + "\n".join(f"- {n}" for n in names.values())
        if self.webcams:
            out += "\nWebcams I can look through on request (live only, nothing recorded):\n" + \
                "\n".join(f"- {w.name}" for w in self.webcams.values())
        return out

    def search_history(self, query: str, camera: str | None = None) -> str:
        if not query or not query.strip():
            return "Please provide something to search for."
        name = self._who(query)
        if name:
            return self.find_sightings(who=name, camera=camera, when=query)
        cams = [self._cam_id(camera)] if camera else None
        res = self.searcher.search(query.strip(), cams)
        rows = res.get("results") or []
        if not rows:
            expl = (res.get("filter") or {}).get("explanation")
            return f"Nothing found for {query!r}." + (f" ({expl})" if expl else "")
        out = [f"Found {len(rows)} match(es) for {query!r}:"]
        for r in rows[:8]:
            label = (r.get("label") or ", ".join(r.get("labels") or [])
                     or ", ".join(t["label"] for t in r.get("tags", [])) or "object")
            ts = r.get("first_ts") or r.get("start_ts")
            line = f"- {label} on {self._cam_name(r.get('camera_id'))}, {_ago(ts)}"
            plate = r.get("plate") or (", ".join(r.get("plates") or []))
            if plate:
                line += f", plate {plate}"
            out.append(line)
        return "\n".join(out)

    # ---- names (people, pets, cars) and object words, matched the way people say them ---------------------
    LABEL_WORDS = {"dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog", "cat": "cat", "cats": "cat",
                   "kitty": "cat", "person": "person", "people": "person", "someone": "person", "anyone": "person",
                   "anybody": "person", "visitor": "person", "visitors": "person", "man": "person", "woman": "person",
                   "car": "car", "cars": "car", "vehicle": "car", "vehicles": "car", "truck": "truck", "trucks": "truck",
                   "bike": "bicycle", "bicycle": "bicycle", "motorcycle": "motorcycle", "bird": "bird", "birds": "bird",
                   "bear": "bear"}

    def _names(self) -> dict[str, tuple[str, str]]:
        """canonical key -> (name, kind) for every named person, pet and car."""
        out = {}
        for kind, reg in (("person", getattr(self.hub, "people", None)), ("pet", getattr(self.hub, "pets", None)),
                          ("vehicle", getattr(self.hub, "cars", None))):
            if reg is not None:
                for n in reg.names().values():
                    out[_canon(n)] = (n, kind)
        return out

    def _who(self, text: str | None) -> str | None:
        """A known name said any way ("bear bear", "Bear-Bear", "bucko" for Buck-O) -> its exact name."""
        if not text:
            return None
        names = self._names()
        key = _canon(text)
        if key in names:
            return names[key][0]
        words = re.findall(r"[a-z0-9]+", text.lower())
        for n in (3, 2, 1):                      # longest phrase first, whole words only
            for i in range(len(words) - n + 1):
                k = "".join(words[i:i + n])
                k2 = k[:-1] if k.endswith("s") and k[:-1] in names else k     # "bucko's"
                if k2 in names:
                    return names[k2][0]
        # Speech-to-text spellings: "Brandy" for Brandi, "Bucco" for Buck-O, "Raylan" for Rylan.
        import difflib
        for n in (3, 2, 1):
            for i in range(len(words) - n + 1):
                k = "".join(words[i:i + n])
                if len(k) < 4:
                    continue
                close = difflib.get_close_matches(k, list(names), n=1, cutoff=0.8)
                if close:
                    return names[close[0]][0]
        return None

    def _label(self, text: str | None) -> str | None:
        if not text:
            return None
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in self.LABEL_WORDS:
                return self.LABEL_WORDS[w]
        return None

    def _window(self, when: str | None, hours: float | None) -> tuple[float, float, str]:
        """(start, end, how it was understood) from words like 'this morning' or 'yesterday at 5pm'."""
        now = time.time()
        if when:
            from .timeparse import parse
            tr = parse(when)
            if tr.start or tr.end:
                st = tr.start.timestamp() if tr.start else 0.0
                en = tr.end.timestamp() if tr.end else now
                return st, min(en, now), tr.label or when
        h = float(hours or 24)
        return now - h * 3600, now, f"the last {h:g}h"

    def _sightings(self, who=None, what=None, camera=None, when=None, hours=None, limit=60, need_clip=False):
        """Matching detections, newest first, each with its clip (if recorded) and where in the clip to start."""
        st, en, span = self._window(when, hours)
        where, args = ["d.first_ts >= ?", "d.first_ts <= ?"], [st, en]
        if who:
            where.append("d.who = ?")
            args.append(who)
        if what:
            where.append("d.label = ?")
            args.append(what)
        if camera:
            where.append("d.camera_id = ?")
            args.append(self._cam_id(camera))
        rows = self.db.all(
            "SELECT d.id, d.camera_id, d.label, d.who, d.first_ts, d.last_ts, d.plate, d.event_id AS ev "
            f"FROM detections d WHERE {' AND '.join(where)} ORDER BY d.first_ts DESC LIMIT ?", args + [int(limit)])
        missing = [r["id"] for r in rows if not r["ev"]]
        if missing:                                   # clips linked the other way, looked up for just these rows
            links = self.db.all(
                f"SELECT ref_id, event_id FROM event_links WHERE kind='detection' AND ref_id IN ({','.join('?' * len(missing))})",
                missing)
            linked = {l["ref_id"]: l["event_id"] for l in links}
            for r in rows:
                r["ev"] = r["ev"] or linked.get(r["id"])
        out = []
        for r in rows:
            e = self.db.one("SELECT clip_path, clip_offset, start_ts FROM events WHERE id = ?", (r["ev"],)) if r["ev"] else None
            r["clip"] = bool(e and e.get("clip_path") and (self.hub.data / e["clip_path"]).exists())
            r["seek"] = max(0.0, e["clip_offset"] + r["first_ts"] - e["start_ts"] - 1.0) if r["clip"] else None
            if need_clip and not r["clip"]:
                continue
            out.append(r)
        return out, span

    @staticmethod
    def _visits(rows: list[dict], gap: float = 180) -> list[dict]:
        """Collapse back-to-back sightings of the same thing on the same camera into one visit."""
        visits: list[dict] = []
        for r in rows:                                   # newest first
            v = visits[-1] if visits else None
            if v and v["camera_id"] == r["camera_id"] and v["who"] == r["who"] and v["label"] == r["label"] \
                    and v["start"] - r["last_ts"] < gap:
                v["start"] = r["first_ts"]
                v["n"] += 1
                v["clip"] = v["clip"] or r["clip"]
                continue
            visits.append({**r, "start": r["first_ts"], "n": 1})
        return visits

    def _their_car(self, name: str | None) -> str | None:
        """Brandi -> "Brandi's Car": comings and goings often show up as the car, not the face."""
        if not name:
            return None
        kinds = {n: k for n, k in self._names().values()}
        if kinds.get(name) != "person":
            return None
        return next((n for n, k in kinds.items() if k == "vehicle" and _canon(n).startswith(_canon(name))), None)

    def _describe(self, r: dict) -> str:
        what = r["who"] + (f" ({r['label']})" if r["label"] not in ("person", "car") else "") if r.get("who") else r["label"]
        return f"{what} on {self._cam_name(r['camera_id'])}, {_ago(r['start'] if 'start' in r else r['first_ts'])}"

    def find_sightings(self, who: str | None = None, what: str | None = None, camera: str | None = None,
                       when: str | None = None, hours: float | None = None, **_ignored: Any) -> str:
        name = self._who(who) or self._who(what)
        label = None if name else (self._label(what) or self._label(who))
        if (who or what) and not name and not label:
            known = ", ".join(n for n, _ in self._names().values())
            return f"I don't know {who or what!r}. Known names: {known}."
        self.last_query = {"ts": time.time(), "who": name, "label": label, "camera": camera, "when": when}
        rows, span = self._sightings(name, label, camera, when, hours)
        thing = name or (label + "s" if label else "anything")
        scope = f" on {self._cam_name(self._cam_id(camera))}" if camera else ""
        car = self._their_car(name)
        if not rows and car:
            crow, _ = self._sightings(car, None, camera, when, hours)
            if crow:
                self.last_query.update(who=car)
                visits = self._visits(crow)
                return (f"I didn't recognise {name} in person{scope} in {span}, but {car} was seen "
                        f"(probably {name} arriving or leaving):\n"
                        + "\n".join(f"- {self._describe(v)}" + (" [video]" if v["clip"] else "") for v in visits[:6]))
        if not rows:
            return f"No sightings of {thing}{scope} in {span}."
        visits = self._visits(rows)
        out = [f"{thing}{scope} in {span}: {len(visits)} visit(s), most recent first:"]
        for v in visits[:8]:
            out.append(f"- {self._describe(v)}" + (" [video]" if v["clip"] else ""))
        out.append("To play one of these, call show_recording with the same who/what/camera/when.")
        return "\n".join(out)

    def show_recording(self, who: str | None = None, what: str | None = None, camera: str | None = None,
                       when: str | None = None, which: str = "latest", **_ignored: Any) -> str:
        name = self._who(who) or self._who(what)
        label = None if name else (self._label(what) or self._label(who))
        lq = self.last_query
        if not (name or label or camera) and time.time() - lq.get("ts", 0) < 600:
            # "show me the video" right after a search: play what that search found
            name, label, camera, when = lq["who"], lq["label"], camera or lq["camera"], when or lq["when"]
        if not (name or label or camera or when):
            return "Which video? Tell me who or what (for example Bear-Bear, a delivery, a car) and roughly when."
        rows, span = self._sightings(name, label, camera, when, None if when else 24 * 7, limit=200, need_clip=True)
        car = self._their_car(name)
        note = ""
        if not rows and car:
            rows, span = self._sightings(car, None, camera, when, None if when else 24 * 7, limit=200, need_clip=True)
            if rows:
                note = f"I didn't recognise {name} in person, so this is {car} instead. "
                name = car
        thing = name or (label or "anything")
        scope = f" on {self._cam_name(self._cam_id(camera))}" if camera else ""
        if not rows:
            return f"I have no recorded video of {thing}{scope} in {span}."
        visits = [v for v in self._visits(rows) if v["clip"]]
        pick = visits[-1] if (which or "").lower().startswith(("first", "earliest", "oldest")) else visits[0]
        playlist = [pick] + [v for v in visits if v is not pick][:5]
        self.last_shown = {"ts": time.time(), "ids": [v["id"] for v in playlist], "title": f"{thing}{scope}"}
        more = len(visits) - 1
        return (note + f"Playing the video of {self._describe(pick)} on the dashboard now."
                + (f" The {min(more, 5)} visit(s) before it are lined up below it ({more} recorded in {span} in all)."
                   if more else ""))

    def recent_activity(self, camera: str | None = None, label: str | None = None,
                        hours: float = 24, limit: int = 15, who: str | None = None, what: str | None = None,
                        when: str | None = None, **_ignored: Any) -> str:
        # Small models mix up the arguments of the two tools: take theirs rather than fail.
        if who or when or self._who(label) or self._who(what):   # "bear bear" is a name, not a detector label
            return self.find_sightings(who=who or label, what=what, camera=camera, when=when, hours=hours)
        label = label or what
        label = self._label(label) or label
        where, args = ["first_ts > ?"], [dt.datetime.now().timestamp() - float(hours) * 3600]
        if camera:
            where.append("camera_id = ?")
            args.append(self._cam_id(camera))
        if label:
            where.append("LOWER(label) = ?")
            args.append(label.lower())
        limit = max(1, min(int(limit), 50))
        rows = self.db.all(
            f"SELECT camera_id, label, confidence, first_ts, plate, who FROM detections "
            f"WHERE {' AND '.join(where)} ORDER BY first_ts DESC LIMIT ?", args + [limit])
        if not rows:
            scope = f" on {self._cam_name(self._cam_id(camera))}" if camera else ""
            what = f" {label}" if label else ""
            return f"No{what} detections{scope} in the last {hours:g}h."
        out = [f"{len(rows)} recent detection(s) in the last {hours:g}h:"]
        for r in rows:
            who = f" ({r['who']})" if r.get("who") else ""
            line = f"- {r['label']}{who} on {self._cam_name(r['camera_id'])}, {_ago(r['first_ts'])}"
            if r.get("plate"):
                line += f", plate {r['plate']}"
            out.append(line)
        # Who it was, by name, over the whole window (not just the rows above), so "has Bucko been out" can be
        # answered from here too.
        named = self.db.all(
            f"SELECT who, camera_id, MAX(first_ts) AS ts, COUNT(*) AS n FROM detections "
            f"WHERE {' AND '.join(where)} AND who IS NOT NULL GROUP BY who ORDER BY ts DESC LIMIT 10", args)
        if named:
            out.append("By name: " + "; ".join(f"{r['who']} last seen {_ago(r['ts'])} on {self._cam_name(r['camera_id'])}"
                                                for r in named) + ".")
        # Every family pet, by name, even ones not seen in this window (so two dogs are never mistaken for one).
        if label in ("dog", "cat", None):
            seen = {r["who"] for r in named}
            for pet in sorted({n for n, k in self._names().values() if k == "pet"} - seen):
                last = self.db.one("SELECT camera_id, MAX(first_ts) AS ts FROM detections WHERE who = ?", (pet,))
                when = f"last seen {_ago(last['ts'])} on {self._cam_name(last['camera_id'])}" if last and last["ts"] else "never seen"
                out.append(f"{pet}: NOT seen in the last {hours:g}h ({when}).")
        return "\n".join(out)

    def look_at_camera(self, camera: str) -> str:
        cid = self._cam_id(camera)
        if cid in self.webcams:
            return self._look_webcam(cid)
        if cid not in self.hub.cams:
            return f"There's no camera called {camera!r}. Cameras: " + ", ".join(self._all_names().values())
        name = self._cam_name(cid)
        res = self.vision.look(cid)
        people = [o for o in res["objects"] if o["label"] == "person"]
        vehicles = [o for o in res["objects"] if o["label"] != "person"]
        out = [f"{name} camera, right now (it is now on the dashboard):"]
        if not people:
            out.append("- Nobody is in view.")
        for o in people:
            if o["name"]:
                out.append(f"- A person: {o['name']} (recognised, {o['why']})")
            elif o.get("maybe"):
                out.append(f"- A person who might be {o['maybe']}, but I'm not sure ({o['why']}).")
            else:
                out.append(f"- A person I can't identify ({o['why']}).")
        for o in vehicles:
            state = "parked" if o["parked"] else "moving"
            if o["name"]:
                out.append(f"- {o['name']} ({state}, {o['why']})")
            elif o.get("maybe"):
                out.append(f"- A {o['label']} that is probably {o['maybe']} ({state}, {o['why']}).")
            else:
                out.append(f"- An unrecognised {o['label']} ({state}).")
        seen = [r for r in res["recent"] if r["label"] == "person" and
                r["who"] not in {o["name"] for o in people}]
        if seen:
            out.append("Recently seen here: " + ", ".join(f"{r['who']} {_ago(r['ts'])}" for r in seen))
        return "\n".join(out)

    def _look_webcam(self, cid: str) -> str:
        res = self.webcam_look(cid)
        name = self._cam_name(cid)
        if res["error"]:
            return f"I couldn't open the {name} webcam: {res['error']}. It may be in use by a video call."
        out = [f"{name} webcam, right now (it is now on the dashboard). It only sees faces, not cars or animals:"]
        if not res["objects"]:
            out.append("- No face in view.")
        for o in res["objects"]:
            if o["name"]:
                out.append(f"- {o['name']} (recognised, {o['why']})")
            elif o.get("maybe"):
                out.append(f"- Someone who might be {o['maybe']}, but I'm not sure ({o['why']}).")
            else:
                out.append(f"- A person I can't identify ({o['why']}).")
        return "\n".join(out)

    def show_cameras(self, cameras: Any = None) -> str:
        if isinstance(cameras, str):                 # small models sometimes send "Driveway, Front Porch"
            import re
            cameras = [c for c in re.split(r"\s*(?:,|\band\b|&)\s*", cameras) if c.strip()]
        cameras = [str(c).strip() for c in (cameras or []) if str(c).strip()]
        if not cameras:
            return "Which cameras? Cameras: " + ", ".join(self._all_names().values())
        parts = [self.look_at_camera(c) for c in cameras[:6]]
        return "\n\n".join(p.replace(" (it is now on the dashboard)", "") for p in parts) + \
            "\n(All of these cameras are now on the dashboard, side by side.)"

    def parked_cars(self, camera: str | None = None) -> str:
        camera = None          # always every camera: small models kept narrowing it to the wrong one
        cams = list(self.hub.cams)
        found = []
        for cid in cams:
            if cid not in self.hub.cams:
                continue
            for o in self.vision.look(cid)["objects"]:
                if o["label"] != "person":
                    found.append((self._cam_name(cid), o))
        if not found:
            return "No vehicles are in view on " + (self._cam_name(cams[0]) if camera else "any camera") + " right now."
        mine = {}
        others = []
        for cam_name, o in found:          # the same car can be in view of two cameras
            if o["name"]:
                mine.setdefault(o["name"], []).append((cam_name, o))
            else:
                others.append((cam_name, o))
        out = ["Vehicles in view right now:"]
        for nm, where in mine.items():
            state = "parked" if any(o["parked"] for _, o in where) else "moving"
            out.append(f"- {nm}: {state}, seen on {', '.join(sorted({c for c, _ in where}))}")
        for cam_name, o in others:
            state = "parked" if o["parked"] else "moving"
            if o.get("maybe"):
                out.append(f"- a {o['label']} on {cam_name} that is probably {o['maybe']} ({state}, {o['why']})")
            else:
                out.append(f"- an unrecognised {o['label']} on {cam_name} ({state})")
        return "\n".join(out)

    def watched_spots(self, what: str = "") -> str:
        w = getattr(self.hub, "watch", None)
        if not w:
            return "No watched spots: picture search (CLIP) is off."
        spots = w.status()
        if what:
            key = _canon(what)
            hit = [z for z in spots if key and (key in _canon(z["what"]) or key in _canon(z["name"])
                                                or _canon(z["what"]) in key or _canon(z["name"]) in key)]
            spots = hit or spots
        if not spots:
            return "No watched spots are set up (make one in Zones with the type 'Watch for something that stays')."
        out = []
        for z in spots:
            cam = self._cam_name(z["camera_id"])
            if z["state"] is None:
                line = f"- {z['what']} ({cam}): not known yet"
                if not (z["counts"]["there"] and z["counts"]["gone"]):
                    line += " (it hasn't been taught what 'there' and 'gone' look like)"
            else:
                word = "there (out)" if z["state"] == "there" else "not there"
                line = f"- {z['what']} ({cam}): {word} since {_ago(z['since'])}"
            if z["blocked"]:
                line += "; can't see the spot right now (something in front of it), so that's the last thing seen"
            out.append(line)
        return "\n".join(out)

    def known_vehicles(self, status: str = "all", limit: int = 15) -> str:
        status = status if status in ("new", "known", "flagged", "all") else "all"
        where, args = ("", []) if status == "all" else ("WHERE status = ?", [status])
        limit = max(1, min(int(limit), 50))
        rows = self.db.all(
            f"SELECT label, plate, visits, last_ts, status, note FROM vehicles {where} "
            f"ORDER BY status = 'flagged' DESC, visits DESC, last_ts DESC LIMIT ?", args + [limit])
        if not rows:
            return f"No {status if status != 'all' else ''} vehicles on record.".replace("  ", " ")
        out = [f"{len(rows)} vehicle(s) ({status}):"]
        for r in rows:
            bits = [r.get("label") or "vehicle"]
            if r.get("plate"):
                bits.append(f"plate {r['plate']}")
            bits.append(f"{r.get('visits', 0)} visit(s)")
            bits.append(f"last seen {_ago(r.get('last_ts'))}")
            if r.get("status") == "flagged":
                bits.append("FLAGGED" + (f": {r['note']}" if r.get("note") else ""))
            out.append("- " + ", ".join(bits))
        return "\n".join(out)


TOOL_SCHEMAS = [
    {"name": "list_cameras",
     "description": "List the cameras TheEye watches, by name. Call this first if unsure which camera a "
                    "question refers to.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "search_history",
     "description": "Search everything TheEye has seen using a natural-language question, e.g. 'delivery "
                    "truck today', 'person at the front door this morning', 'silver SUV'. Optionally limit to "
                    "one camera by name.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look for, in plain language."},
         "camera": {"type": "string", "description": "Optional camera name to limit the search to."}},
         "required": ["query"]}},
    {"name": "recent_activity",
     "description": "List recent object detections (people, vehicles, animals) in the last N hours, newest "
                    "first. Use for 'what's happened lately' or 'any activity on the driveway'. For a NAMED person, "
                    "dog or car (Bear-Bear, Bucko, Rylan...) use find_sightings instead.",
     "inputSchema": {"type": "object", "properties": {
         "camera": {"type": "string", "description": "Optional camera name."},
         "label": {"type": "string", "description": "Optional object type, e.g. person, car, truck, dog."},
         "hours": {"type": "number", "description": "How far back to look, in hours (default 24)."}}}},
    {"name": "look_at_camera",
     "description": "Look at one camera RIGHT NOW: who is there (named if TheEye recognises their face), which "
                    "vehicles are in view and whose they are. Also puts that camera on the dashboard. Use for "
                    "'who's at the front door', 'is anyone in the back yard', 'show me the driveway'. The front "
                    "door is the 'Front Porch' camera. The desk webcam in the office is the 'Office' camera (faces only).",
     "inputSchema": {"type": "object", "properties": {
         "camera": {"type": "string", "description": "Camera name, e.g. Front Porch, Driveway, Back Yard."}},
         "required": ["camera"]}},
    {"name": "find_sightings",
     "description": "When and where a named person, pet or car (or a kind of thing: dog, cat, person, car, truck) "
                    "was seen: 'when was Bear Bear last in the back yard', 'has Bucko been out today', 'when did "
                    "Rylan's car leave'. Known names: {names}. Spoken names match loosely ('bear bear' = Bear-Bear).",
     "inputSchema": {"type": "object", "properties": {
         "who": {"type": "string", "description": "A known name, e.g. Bear-Bear, Buck-O, Eric, Rylan's Car."},
         "what": {"type": "string", "description": "Or a kind of thing: dog, cat, person, car, truck."},
         "camera": {"type": "string", "description": "The camera, whenever the user names a place: 'backyard' -> "
                                                     "Back Yard, 'front door' -> Front Porch, 'driveway' -> Driveway."},
         "when": {"type": "string", "description": "Optional time in words: 'today', 'this morning', 'yesterday "
                                                   "evening', 'last night'. Default: the last 24 hours."}}}},
    {"name": "show_recording",
     "description": "PLAY a recorded video from the past on the dashboard: 'show me the video of Bear Bear in the "
                    "back yard', 'show me when the delivery came this morning', 'play the last time Rylan got home'. "
                    "Picks the most recent matching recording (or the first one with which='first'). Known names: "
                    "{names}.",
     "inputSchema": {"type": "object", "properties": {
         "who": {"type": "string", "description": "A known name, e.g. Bear-Bear, Buck-O, Brandi."},
         "what": {"type": "string", "description": "Or a kind of thing: dog, cat, person, car, truck."},
         "camera": {"type": "string", "description": "The camera, whenever the user names a place: 'backyard' -> "
                                                     "Back Yard, 'front door' -> Front Porch, 'driveway' -> Driveway."},
         "when": {"type": "string", "description": "Optional time in words, e.g. 'this morning', 'yesterday', "
                                                   "'this week'."},
         "which": {"type": "string", "enum": ["latest", "first"], "description": "Default latest."}}}},
    {"name": "show_cameras",
     "description": "Show SEVERAL cameras at once on the dashboard, side by side, and say who and what is on each "
                    "right now. Use when the user asks for more than one camera, e.g. 'show me the driveway and the "
                    "front porch', 'show me the front and back yard'. Pass every camera they named.",
     "inputSchema": {"type": "object", "properties": {
         "cameras": {"type": "array", "items": {"type": "string"},
                     "description": "Camera names, e.g. [\"Driveway\", \"Front Porch\"]."}},
         "required": ["cameras"]}},
    {"name": "parked_cars",
     "description": "Which vehicles are in view right now across all cameras (or one), and whose they are "
                    "(Eric's, Rylan's, ...) by plate or appearance. Use for 'whose car is outside', 'is Rylan "
                    "home' (his car), 'what cars are parked nearby'. Always checks every camera. For vehicles on "
                    "one particular camera use look_at_camera.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "watched_spots",
     "description": "Whether something that stays put is in its spot right now: 'are the trash cans out?', 'did "
                    "the bins get put out', 'have the cans been brought in'. Says since when, and when the spot is "
                    "hidden (a car parked in front). Give 'what' to pick one spot, e.g. 'trash cans'.",
     "inputSchema": {"type": "object", "properties": {
         "what": {"type": "string", "description": "Optional: which watched thing, e.g. 'trash cans'."}}}},
    {"name": "known_vehicles",
     "description": "List vehicles TheEye has learned, with plates and visit counts. status can be 'flagged' "
                    "(watch-listed), 'known', 'new', or 'all' (default).",
     "inputSchema": {"type": "object", "properties": {
         "status": {"type": "string", "enum": ["new", "known", "flagged", "all"]}}}},
]


def _dispatch(tools: _Tools, body: dict) -> dict | None:
    """Handle one JSON-RPC message. Returns a response dict, or None for notifications."""
    method = body.get("method")
    rid = body.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO}}
    if method == "notifications/initialized" or rid is None:
        return None  # notification: no reply
    if method == "tools/list":
        try:
            names = ", ".join(f"{n} ({'dog/pet' if k == 'pet' else k})" for n, k in tools._names().values()) or "none yet"
        except Exception:  # noqa: BLE001
            names = "see find_sightings"
        schemas = json.loads(json.dumps(TOOL_SCHEMAS).replace("{names}", names.replace('"', "'")))
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": schemas}}
    if method == "tools/call":
        params = body.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = getattr(tools, name, None) if name in {t["name"] for t in TOOL_SCHEMAS} else None
        if fn is None:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"Unknown tool: {name}"}}
        try:
            text = fn(**args)
        except TypeError as e:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(e)}}
        except Exception as e:  # noqa: BLE001 — never crash the assistant; report the failure as content
            log.warning("tool %s failed: %s", name, e)
            return {"jsonrpc": "2.0", "id": rid,
                    "result": {"content": [{"type": "text", "text": f"Tool error: {e}"}], "isError": True}}
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": text}], "isError": False}}
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def _handler_factory(tools: _Tools, token_ok: Callable[[str | None], bool]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence default stderr logging
            pass

        def _send(self, code: int, payload: dict | None):
            data = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(code)
            if data:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if data:
                self.wfile.write(data)

        def do_GET(self):
            """Read-only pictures for the Jarvis dashboard: /cameras.json, /cam/<id>/snapshot.jpg, /cam/<id>/look.json"""
            if not token_ok(self.headers.get("Authorization")):
                return self._send(401, {"error": "unauthorized"})
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            hub = tools.hub
            try:
                if parts == ["cameras.json"]:
                    order = {c.id: i for i, c in enumerate(hub.cfg.cameras)}
                    cams = sorted(hub.cams.values(), key=lambda c: order.get(c.id, len(order)))
                    return self._send(200, {"cameras": [{"id": c.id, "name": c.cfg.name or c.id,
                                                         "live": bool(c.latest_jpeg)} for c in cams] +
                                            [{"id": w.id, "name": w.name, "live": True, "webcam": True}
                                             for w in tools.webcams.values()]})
                if parts == ["recordings", "last.json"]:
                    ls = tools.last_shown
                    return self._send(200, {"ts": ls["ts"], "ids": ls["ids"], "title": ls["title"]})
                if len(parts) == 2 and parts[0] == "recording" and parts[1].split(".")[0].isdigit():
                    return self._recording(int(parts[1].split(".")[0]), parts[1].rsplit(".", 1)[-1])
                if len(parts) == 3 and parts[0] == "cam":
                    cid = tools._cam_id(parts[1])
                    if cid in tools.webcams:
                        return self._webcam(cid, parts[2])
                    cam = hub.cams.get(cid)
                    if cam is None:
                        return self._send(404, {"error": "no such camera"})
                    if parts[2] == "snapshot.jpg":
                        jpg = cam.latest_jpeg
                        if not jpg:
                            return self._send(503, {"error": "no frame yet"})
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(jpg)
                        return None
                    if parts[2] == "live.mp4":
                        return self._live(cam)
                    if parts[2] == "look.json":
                        res = tools.vision.look(cid)
                        res["name"] = tools._cam_name(cid)
                        return self._send(200, res)
            except Exception as e:  # noqa: BLE001
                log.warning("GET %s failed: %s", self.path, e)
                return self._send(500, {"error": str(e)})
            return self._send(404, {"error": "not found"})

        def _webcam(self, cid: str, what: str):
            if what == "snapshot.jpg":
                jpg = tools.webcams[cid].jpeg()
                if not jpg:
                    return self._send(503, {"error": tools.webcams[cid].error or "no frame yet"})
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpg)
                return None
            if what == "look.json":
                res = tools.webcam_look(cid)
                res["name"] = tools._cam_name(cid)
                return self._send(200, res)
            if what == "live.mp4":
                return self._webcam_live(tools.webcams[cid])
            if what == "frames.mjpeg":
                return self._webcam_frames(tools.webcams[cid])
            return self._send(404, {"error": "not found"})

        def _webcam_frames(self, w):
            """Every webcam frame as a 640-px JPEG in a multipart stream: for a local gesture recognizer. Keeps the
            webcam open while connected."""
            import cv2
            if w.grab(full=False) is None:
                return self._send(503, {"error": w.error or "webcam not available"})
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            ts = 0.0
            try:
                while True:
                    img, ts2 = w.next_frame(ts, full=False)
                    if img is None:
                        if w.error is not None or w.grab(full=False) is None:
                            break
                        continue
                    ts = ts2
                    h, wd = img.shape[:2]
                    small = img if wd <= 640 else cv2.resize(img, (640, int(h * 640 / wd)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if not ok:
                        continue
                    b = buf.tobytes()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\nX-Ts: %.3f\r\n\r\n"
                                     % (len(b), ts) + b + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return None

        def _webcam_live(self, w):
            if w.grab() is None:
                return self._send(503, {"error": w.error or "webcam not available"})
            with _live_lock:
                if _live_count["n"] >= LIVE_MAX:
                    return self._send(503, {"error": "too many live streams"})
                _live_count["n"] += 1
            try:
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("X-Mime", w.LIVE_MIME)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()

                def write(b: bytes) -> None:
                    self.wfile.write(b)
                    self.wfile.flush()
                w.live(write)
            except (BrokenPipeError, ConnectionResetError):
                pass                                    # the viewer went away
            except Exception as e:  # noqa: BLE001
                log.info("webcam live ended: %s", e)
            finally:
                with _live_lock:
                    _live_count["n"] -= 1
            return None

        def _recording(self, det_id: int, ext: str):
            d = tools.db.one("SELECT id, camera_id, label, who, first_ts, last_ts, COALESCE(event_id, (SELECT l.event_id "
                             "FROM event_links l WHERE l.kind='detection' AND l.ref_id=detections.id LIMIT 1)) AS ev "
                             "FROM detections WHERE id = ?", (det_id,))
            e = tools.db.one("SELECT clip_path, clip_offset, start_ts, end_ts FROM events WHERE id = ?", (d["ev"],)) \
                if d and d.get("ev") else None
            data = tools.hub.data.resolve()
            path = (data / e["clip_path"]).resolve() if e and e.get("clip_path") else None
            if not path or data not in path.parents or not path.is_file():
                return self._send(404, {"error": "no recording"})
            if ext == "json":
                return self._send(200, {
                    "id": d["id"], "camera": tools._cam_name(d["camera_id"]), "camera_id": d["camera_id"],
                    "label": d["label"], "who": d["who"], "ts": d["first_ts"], "ago": _ago(d["first_ts"]),
                    "seek": max(0.0, e["clip_offset"] + d["first_ts"] - e["start_ts"] - 1.0),
                    "duration": (e["end_ts"] or d["last_ts"]) - e["start_ts"] + e["clip_offset"] if e.get("end_ts") else None})
            if ext != "mp4":
                return self._send(404, {"error": "not found"})
            size = path.stat().st_size
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m and (m.group(1) or m.group(2)):          # browsers (iOS especially) stream video in byte ranges
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "private, max-age=3600")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(256 * 1024, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    left -= len(chunk)
            return None

        def _live(self, cam):
            """Smooth live video for the dashboard: the camera's own H.264 as an endless fragmented MP4, from TheEye's
            RAM loop (no extra camera connection, no re-encoding) -- the same LiveMP4 TheEye's full view plays."""
            from .recorder import LiveMP4
            if cam.template_stream is None or cam.status.get("main") != "live":
                return self._send(503, {"error": "camera is not streaming"})
            with _live_lock:
                if _live_count["n"] >= LIVE_MAX:
                    return self._send(503, {"error": "too many live streams"})
                _live_count["n"] += 1
            ls = None
            try:
                audio = "audio=1" in (self.path.split("?", 1)[1] if "?" in self.path else "")
                ls = LiveMP4(cam, audio)
                self.close_connection = True
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("X-Mime", ls.mime)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = ls.read(1.0)
                    if chunk is None:
                        break
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                                    # the viewer went away
            finally:
                if ls is not None:
                    ls.close()
                with _live_lock:
                    _live_count["n"] -= 1
            return None

        def _identify(self):
            import cv2
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 8_000_000:
                return self._send(400, {"error": "send a JPEG body"})
            img = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return self._send(400, {"error": "couldn't read that image"})
            return self._send(200, {"faces": tools.vision.identify(img), "size": [img.shape[1], img.shape[0]]})

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") == "/identify":
                if not token_ok(self.headers.get("Authorization")):
                    return self._send(401, {"error": "unauthorized"})
                try:
                    return self._identify()
                except Exception as e:  # noqa: BLE001
                    log.warning("identify failed: %s", e)
                    return self._send(500, {"error": str(e)})
            if not token_ok(self.headers.get("Authorization")):
                return self._send(401, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32001, "message": "unauthorized"}})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:  # noqa: BLE001
                return self._send(400, {"jsonrpc": "2.0", "id": None,
                                        "error": {"code": -32700, "message": "parse error"}})
            resp = _dispatch(tools, body)
            if resp is None:
                return self._send(202, None)  # notification acknowledged
            self._send(200, resp)

    return Handler


def serve(host: str, port: int, hub: Any, searcher: Any, db: Any,
          token_ok: Callable[[str | None], bool]) -> ThreadingHTTPServer:
    """Start the loopback MCP server on a daemon thread. Returns the server (call .shutdown() to stop)."""
    tools = _Tools(hub, searcher, db)
    httpd = ThreadingHTTPServer((host, port), _handler_factory(tools, token_ok))
    threading.Thread(target=httpd.serve_forever, name="theeye-mcp", daemon=True).start()
    log.info("read-only MCP server on http://%s:%d (loopback, token-guarded)", host, port)
    return httpd
