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
import time
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

    # id <-> display name, accepting either from the model
    def _cam_id(self, cam: str | None) -> str | None:
        if not cam:
            return None
        names = self.searcher.cameras  # {id: name}
        if cam in names:
            return cam
        for cid, name in names.items():
            if name.lower() == cam.lower():
                return cid
        return cam  # unknown: let the query return nothing rather than guess

    def _cam_name(self, cid: str | None) -> str:
        return self.searcher.cameras.get(cid, cid or "?")

    # ---- tool implementations ------------------------------------------------
    def list_cameras(self) -> str:
        names = self.searcher.cameras
        if not names:
            return "No cameras are configured."
        return "Cameras TheEye watches:\n" + "\n".join(f"- {n}" for n in names.values())

    def search_history(self, query: str, camera: str | None = None) -> str:
        if not query or not query.strip():
            return "Please provide something to search for."
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

    def recent_activity(self, camera: str | None = None, label: str | None = None,
                        hours: float = 24, limit: int = 15) -> str:
        where, args = ["first_ts > ?"], [dt.datetime.now().timestamp() - float(hours) * 3600]
        if camera:
            where.append("camera_id = ?")
            args.append(self._cam_id(camera))
        if label:
            where.append("LOWER(label) = ?")
            args.append(label.lower())
        limit = max(1, min(int(limit), 50))
        rows = self.db.all(
            f"SELECT camera_id, label, confidence, first_ts, plate FROM detections "
            f"WHERE {' AND '.join(where)} ORDER BY first_ts DESC LIMIT ?", args + [limit])
        if not rows:
            scope = f" on {self._cam_name(self._cam_id(camera))}" if camera else ""
            what = f" {label}" if label else ""
            return f"No{what} detections{scope} in the last {hours:g}h."
        out = [f"{len(rows)} recent detection(s) in the last {hours:g}h:"]
        for r in rows:
            line = f"- {r['label']} on {self._cam_name(r['camera_id'])}, {_ago(r['first_ts'])}"
            if r.get("plate"):
                line += f", plate {r['plate']}"
            out.append(line)
        return "\n".join(out)

    def look_at_camera(self, camera: str) -> str:
        cid = self._cam_id(camera)
        if cid not in self.hub.cams:
            return f"There's no camera called {camera!r}. Cameras: " + ", ".join(self.searcher.cameras.values())
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

    def show_cameras(self, cameras: Any = None) -> str:
        if isinstance(cameras, str):                 # small models sometimes send "Driveway, Front Porch"
            import re
            cameras = [c for c in re.split(r"\s*(?:,|\band\b|&)\s*", cameras) if c.strip()]
        cameras = [str(c).strip() for c in (cameras or []) if str(c).strip()]
        if not cameras:
            return "Which cameras? Cameras: " + ", ".join(self.searcher.cameras.values())
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
                    "first. Use for 'what's happened lately' or 'any activity on the driveway'.",
     "inputSchema": {"type": "object", "properties": {
         "camera": {"type": "string", "description": "Optional camera name."},
         "label": {"type": "string", "description": "Optional object type, e.g. person, car, truck, dog."},
         "hours": {"type": "number", "description": "How far back to look, in hours (default 24)."}}}},
    {"name": "look_at_camera",
     "description": "Look at one camera RIGHT NOW: who is there (named if TheEye recognises their face), which "
                    "vehicles are in view and whose they are. Also puts that camera on the dashboard. Use for "
                    "'who's at the front door', 'is anyone in the back yard', 'show me the driveway'. The front "
                    "door is the 'Front Porch' camera.",
     "inputSchema": {"type": "object", "properties": {
         "camera": {"type": "string", "description": "Camera name, e.g. Front Porch, Driveway, Back Yard."}},
         "required": ["camera"]}},
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
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOL_SCHEMAS}}
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
                                                         "live": bool(c.latest_jpeg)} for c in cams]})
                if len(parts) == 3 and parts[0] == "cam":
                    cid = tools._cam_id(parts[1])
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
                    if parts[2] == "look.json":
                        res = tools.vision.look(cid)
                        res["name"] = tools._cam_name(cid)
                        return self._send(200, res)
            except Exception as e:  # noqa: BLE001
                log.warning("GET %s failed: %s", self.path, e)
                return self._send(500, {"error": str(e)})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
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
