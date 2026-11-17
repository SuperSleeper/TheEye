"""HTTP API + static web UI."""
from __future__ import annotations

import asyncio
import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, configapi
from .config import save_config
from .db import jload
from .hub import LOG_MAX_BYTES, MOTION_KEEP_DAYS, SOUND_KEEP_DAYS, Hub
from .search import Searcher


def _du(root: Path) -> tuple[int, int, Optional[float]]:
    """(bytes, files, oldest mtime) under root."""
    size = n = 0
    oldest: Optional[float] = None
    for p in root.rglob("*"):
        try:
            st = p.stat()
        except OSError:                  # deleted by retention while walking
            continue
        if p.is_file():
            size += st.st_size
            n += 1
            oldest = st.st_mtime if oldest is None else min(oldest, st.st_mtime)
    return size, n, oldest


def _storage_scan(data: Path, hub: Hub, install: Path) -> dict:
    import shutil
    names = {c.id: c.name or c.id for c in hub.cfg.cameras}
    ids = set(names)
    for sub in ("clips", "snaps"):
        if (data / sub).is_dir():
            ids |= {p.name for p in (data / sub).iterdir() if p.is_dir()}
    cams = []
    for cid in ids:
        clips, nclips, oldest = _du(data / "clips" / cid)
        snaps, nsnaps, oldest_snap = _du(data / "snaps" / cid)
        cam = hub.cams.get(cid)
        cams.append({"id": cid, "name": names.get(cid, "Face pictures (People)" if cid == "faces" else cid),
                     "configured": cid in names, "special": cid == "faces" and cid not in names,
                     "clips_bytes": clips, "clips": nclips, "oldest_clip": oldest,
                     "snaps_bytes": snaps, "snaps": nsnaps, "oldest_snap": oldest_snap, "total_bytes": clips + snaps,
                     "ram_bytes": cam.ring.stats["bytes"] if cam else 0})
    quotas = hub.cfg.storage.clip_quotas([c.id for c in hub.cfg.cameras])
    for r in cams:
        r["quota_bytes"] = quotas.get(r["id"])
    cams.sort(key=lambda r: -r["total_bytes"])
    db_bytes = sum(p.stat().st_size for p in data.glob("theeye.db*") if p.is_file())
    files = sum(r["total_bytes"] for r in cams)
    other = sum(p.stat().st_size for p in data.iterdir() if p.is_file() and not p.name.startswith("theeye.db"))
    disk = shutil.disk_usage(data)
    logs = [p for p in install.glob("*.log*") if p.is_file()]
    oldest = lambda k: min((r[k] for r in cams if r[k]), default=None)        # noqa: E731
    count = lambda sql: hub.db.one(sql)["n"]                                   # noqa: E731
    return {"time": time.time(), "path": str(data), "cameras": cams, "database_bytes": db_bytes,
            "other_bytes": other, "clips_bytes": sum(r["clips_bytes"] for r in cams),
            "clips": sum(r["clips"] for r in cams), "oldest_clip": oldest("oldest_clip"),
            "snaps_bytes": sum(r["snaps_bytes"] for r in cams), "snaps": sum(r["snaps"] for r in cams),
            "oldest_snap": oldest("oldest_snap"),
            "rows": {"events": count("SELECT COUNT(*) AS n FROM events"),
                     "detections": count("SELECT COUNT(*) AS n FROM detections"),
                     "searchable": count("SELECT COUNT(*) AS n FROM detections WHERE embedding IS NOT NULL")},
            "logs_bytes": sum(p.stat().st_size for p in logs), "logs": len(logs),
            "log_limit_bytes": LOG_MAX_BYTES,
            "ram_bytes": sum(r["ram_bytes"] for r in cams), "ram_budget_bytes": hub.cfg.storage.ram_budget_gb * 1e9,
            "keep_days": {"motion": MOTION_KEEP_DAYS, "sound": SOUND_KEEP_DAYS},
            "total_bytes": files + db_bytes + other, "clip_limit_bytes": hub.cfg.storage.max_gb * 1e9,
            "disk_total": disk.total, "disk_free": disk.free}

WEB = Path(__file__).parent / "web"

# Live video tiles: each stream blocks a thread while it waits for packets, so they get their own pool
# instead of starving the default one that the rest of the API uses.
LIVE_POOL = ThreadPoolExecutor(max_workers=64, thread_name_prefix="live-tile")
LIVE_MAX_STREAMS = 48            # across all viewers; beyond this, extra tiles stay on snapshots
LIVE_SESSION_CHECK_S = 30        # re-check the sign-in this often, so signing out stops open streams


class ZoneIn(BaseModel):
    camera_id: str
    name: str
    points: list[list[float]]
    kind: str = Field("motion", pattern="^(motion|detect|exclude)$")
    sensitivity: float = Field(0.5, ge=0, le=1)
    min_area: float = Field(0.02, ge=0, le=1)
    min_frames: int = Field(2, ge=1, le=100)
    color: str = "#ff3b30"
    enabled: bool = True
    ai: bool = True


class RuleIn(BaseModel):
    name: str
    camera_id: str = "*"
    enabled: bool = True
    match: str = Field("any", pattern="^(any|all)$")
    conditions: list[dict[str, Any]]
    preroll: float = Field(10, ge=0, le=3600)
    postroll: float = Field(10, ge=0, le=3600)
    max_length: float = Field(600, ge=5, le=86400)
    schedule: Optional[dict] = None
    record: bool = True
    webhook: Optional[str] = None


class ManualIn(BaseModel):
    camera_id: str
    duration: float = Field(30, ge=1, le=3600)
    preroll: float = Field(10, ge=0, le=3600)
    note: str = ""


def create_app(hub: Hub, searcher: Searcher, config_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        hub.start()
        try:
            yield
        finally:
            await asyncio.to_thread(hub.stop)

    app = FastAPI(title="TheEye NVR", lifespan=lifespan)
    db = hub.db
    data = hub.data.resolve()
    auth_store = auth.AuthStore(config_path.parent / "auth.json", db)
    live_streams = {"n": 0}

    def cam_or_404(cam_id: str):
        c = hub.cams.get(cam_id)
        if not c:
            raise HTTPException(404, "camera not found")
        return c

    # ------------------------------------------------------------- cameras
    @app.get("/api/cameras")
    def cameras():
        order = {c.id: i for i, c in enumerate(hub.cfg.cameras)}   # config order = display order
        cams = sorted(hub.cams.values(), key=lambda c: order.get(c.id, len(order)))
        return [{"id": c.id, "name": c.cfg.name or c.id, "status": c.status, "ring": c.ring.stats,
                 "lag": c.lag_now(), "feed": "sub" if c.cfg.sub_url else "main",
                 "ring_seconds": c.cfg.ring_seconds, "has_audio": c.has_audio,
                 "sound_db": c.sound_db if time.time() - c.sound_ts < 2 else None,
                 "sound_threshold_db": c.cfg.sound_threshold_db if c.cfg.sound_detect else None} for c in cams]

    @app.get("/api/sound")
    def sound(camera_id: Optional[str] = None, hours: float = Query(6, gt=0, le=168)):
        """Loudness history per camera (peak and average dBFS), bucketed so a chart gets ~360 points."""
        since = time.time() - hours * 3600
        bucket = max(10.0, hours * 3600 / 360)
        where, args = "ts >= ?", [since]
        if camera_id:
            where += " AND camera_id = ?"
            args.append(camera_id)
        rows = db.all(f"SELECT camera_id, CAST(ts / {bucket} AS INTEGER) * {bucket} AS t, MAX(peak_db) AS peak,"
                      f" AVG(avg_db) AS avg FROM sound_levels WHERE {where} GROUP BY camera_id, t ORDER BY t", args)
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["camera_id"], []).append([r["t"], round(r["peak"], 1), round(r["avg"], 1)])
        return {"since": since, "bucket": bucket, "cameras": out}

    @app.get("/api/cameras/{cam_id}/snapshot.jpg")
    def snapshot(cam_id: str):
        c = cam_or_404(cam_id)
        if not c.latest_jpeg:
            raise HTTPException(503, "no frame yet")
        # How old this picture is right now: lag when it arrived + time since (the browser adds its fetch time).
        age = time.time() - c.latest_frame_ts + c.latest_frame_lag
        return Response(c.latest_jpeg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Frame-Age": f"{age:.3f}"})

    @app.get("/api/cameras/{cam_id}/live.mp4")
    async def live_mp4(cam_id: str, request: Request, audio: bool = True):
        """Full-resolution live stream (H.264 copied, audio as AAC) for the full-window view."""
        from .recorder import LiveMP4
        c = cam_or_404(cam_id)
        if c.template_stream is None or c.status.get("main") != "live":
            raise HTTPException(503, "camera is not streaming")
        ls = await asyncio.to_thread(LiveMP4, c, audio)

        async def gen():
            try:
                while True:
                    chunk = await asyncio.to_thread(ls.read, 1.0)
                    if chunk is None or await request.is_disconnected():
                        break
                    if chunk:
                        yield chunk
            finally:
                ls.close()
        return StreamingResponse(gen(), media_type="video/mp4",
                                 headers={"Cache-Control": "no-store", "X-Mime": ls.mime})

    @app.websocket("/api/live/ws")
    async def live_ws(ws: WebSocket):
        """Live video for the tiles: every camera's H.264 as fMP4 over ONE WebSocket (browsers allow only ~6
        HTTP connections per site, so one stream per tile would block the rest of the page).

        Client -> {"cams": [ids]} picks the cameras to send; {"ping": t} -> {"pong": t}.
        Server -> {"cam", "mime"} before a (re)started stream, {"cam", "end": true} when one stops,
                  and binary frames: [u8 id length][id][f64 age seconds][fMP4 bytes], where age is how far
                  behind real time the newest frame in the chunk is.
        """
        # The auth middleware only sees HTTP requests, so a WebSocket checks the session itself, and must
        # come from this site (a browser sends cookies on cross-site WebSockets too).
        https = ws.url.scheme in ("wss", "https")
        origin, host = ws.headers.get("origin", ""), ws.headers.get("host", "")
        token = ws.cookies.get(auth.cookie_name(https))
        if not origin or urlsplit(origin).netloc.lower() != host.lower() or not auth_store.session(token):
            await ws.close(code=1008)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()
        send_lock = asyncio.Lock()
        tasks: dict[str, asyncio.Task] = {}

        async def send(kind: str, payload) -> None:
            async with send_lock:
                await (ws.send_bytes(payload) if kind == "b" else ws.send_json(payload))

        async def stream(cam_id: str) -> None:
            from .recorder import LiveMP4
            c = hub.cams.get(cam_id)
            tag = cam_id.encode()[:255]
            while c is not None:
                if c.template_stream is None or c.status.get("main") != "live":
                    await asyncio.sleep(2)
                    continue
                if live_streams["n"] >= LIVE_MAX_STREAMS:
                    await send("j", {"cam": cam_id, "end": True, "why": "busy"})
                    return
                live_streams["n"] += 1
                ls = None
                try:
                    ls = await loop.run_in_executor(LIVE_POOL, LiveMP4, c, False)
                    await send("j", {"cam": cam_id, "mime": ls.mime})
                    while True:
                        chunk = await loop.run_in_executor(LIVE_POOL, ls.read, 0.5)
                        if chunk is None:
                            break
                        if chunk:
                            age = time.time() - ls.newest_wall + c.lag["main"].last if ls.newest_wall else 0.0
                            await send("b", struct.pack(f">B{len(tag)}sd", len(tag), tag, age) + chunk)
                    await send("j", {"cam": cam_id, "end": True})
                except (WebSocketDisconnect, RuntimeError):
                    return
                finally:
                    live_streams["n"] -= 1
                    if ls is not None:
                        await loop.run_in_executor(LIVE_POOL, ls.close)
                await asyncio.sleep(1)          # stream ended (camera reconnected, viewer too slow): restart

        def want(ids) -> None:
            ids = {i for i in ids if isinstance(i, str) and i in hub.cams}
            for cid in list(tasks):
                if cid not in ids:
                    tasks.pop(cid).cancel()
            for cid in ids - tasks.keys():
                tasks[cid] = asyncio.create_task(stream(cid))

        async def watch_session() -> None:     # signing out (or the session expiring) ends the stream
            while True:
                await asyncio.sleep(LIVE_SESSION_CHECK_S)
                if not await asyncio.to_thread(auth_store.session, token):
                    await ws.close(code=1008)
                    return

        watcher = asyncio.create_task(watch_session())
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    continue
                if isinstance(msg.get("cams"), list):
                    want(msg["cams"][:64])
                if "ping" in msg:
                    await send("j", {"pong": msg["ping"]})
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            watcher.cancel()
            for t in tasks.values():
                t.cancel()

    @app.get("/api/cameras/{cam_id}/live.mjpg")
    async def live(cam_id: str):
        c = cam_or_404(cam_id)

        async def gen():
            last = 0.0
            while True:
                jpg, last = await asyncio.to_thread(c.wait_jpeg, last, 2.0)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/cameras/{cam_id}/state")
    def cam_state(cam_id: str):
        cam_or_404(cam_id)
        st = hub.cam_state(cam_id)
        return {"motion_zones": sorted(st.motion_zones), "tracks": st.tracks,
                "recording": hub.rec.active().get(cam_id)}

    # --------------------------------------------------------------- zones
    @app.get("/api/zones")
    def zones(camera_id: Optional[str] = None):
        rows = (db.all("SELECT * FROM zones WHERE camera_id=? ORDER BY id", (camera_id,)) if camera_id
                else db.all("SELECT * FROM zones ORDER BY id"))
        for r in rows:
            r["points"] = jload(r["points"], [])
            r["enabled"] = bool(r["enabled"])
            r["ai"] = bool(r.get("ai", 1))
        return rows

    def _zone_params(z: ZoneIn):
        if len(z.points) < 3:
            raise HTTPException(400, "zone needs at least 3 points")
        pts = [[min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in z.points]
        return (z.camera_id, z.name, json.dumps(pts), z.kind, z.sensitivity, z.min_area, z.min_frames,
                z.color, int(z.enabled), int(z.ai))

    @app.post("/api/zones")
    def zone_create(z: ZoneIn):
        cam_or_404(z.camera_id)
        zid = db.insert("INSERT INTO zones(camera_id,name,points,kind,sensitivity,min_area,min_frames,color,enabled,ai)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)", _zone_params(z))
        hub.reload_zones()
        return {"id": zid}

    @app.put("/api/zones/{zid}")
    def zone_update(zid: int, z: ZoneIn):
        db.execute("UPDATE zones SET camera_id=?,name=?,points=?,kind=?,sensitivity=?,min_area=?,min_frames=?,"
                   "color=?,enabled=?,ai=? WHERE id=?", _zone_params(z) + (zid,))
        hub.reload_zones()
        return {"id": zid}

    @app.delete("/api/zones/{zid}")
    def zone_delete(zid: int):
        db.execute("DELETE FROM zones WHERE id=?", (zid,))
        hub.reload_zones()
        return {"ok": True}

    # --------------------------------------------------------------- rules
    @app.get("/api/rules")
    def rules():
        rows = db.all("SELECT * FROM rules ORDER BY id")
        for r in rows:
            r["conditions"], r["schedule"] = jload(r["conditions"], []), jload(r["schedule"])
            r["enabled"], r["record"] = bool(r["enabled"]), bool(r["record"])
        return rows

    def _rule_params(r: RuleIn):
        if not r.conditions:
            raise HTTPException(400, "rule needs at least one condition")
        return (r.name, r.camera_id, int(r.enabled), r.match, json.dumps(r.conditions), r.preroll, r.postroll,
                r.max_length, json.dumps(r.schedule) if r.schedule else None, int(r.record), r.webhook or None)

    @app.post("/api/rules")
    def rule_create(r: RuleIn):
        rid = db.insert("INSERT INTO rules(name,camera_id,enabled,match,conditions,preroll,postroll,max_length,"
                        "schedule,record,webhook) VALUES(?,?,?,?,?,?,?,?,?,?,?)", _rule_params(r))
        hub.reload_rules()
        return {"id": rid}

    @app.put("/api/rules/{rid}")
    def rule_update(rid: int, r: RuleIn):
        db.execute("UPDATE rules SET name=?,camera_id=?,enabled=?,match=?,conditions=?,preroll=?,postroll=?,"
                   "max_length=?,schedule=?,record=?,webhook=? WHERE id=?", _rule_params(r) + (rid,))
        hub.reload_rules()
        return {"id": rid}

    @app.delete("/api/rules/{rid}")
    def rule_delete(rid: int):
        db.execute("DELETE FROM rules WHERE id=?", (rid,))
        hub.reload_rules()
        return {"ok": True}

    # -------------------------------------------------------------- events
    def _play_lead_in(e: dict, dets: list[dict]) -> float:
        """Seconds into the event's pre-roll where activity begins: the first motion/object that started
        inside the clip window (things already in view before it, like parked cars, don't count), else the
        moment the rule fired. Starts 1 s early for context."""
        end = e["end_ts"] or time.time()
        starts = [s for s in [m["start_ts"] for m in e["motion"]] + [d["first_ts"] for d in dets]
                  if e["start_ts"] <= s <= end]
        trig = e.get("trigger_ts")
        if trig is None and e["rule_id"]:              # events from before trigger_ts was stored
            trig = e["start_ts"] + ((db.one("SELECT preroll FROM rules WHERE id=?", (e["rule_id"],)) or {})
                                    .get("preroll") or 0)
        anchor = min(starts + ([trig] if trig else [])) if (starts or trig) else e["start_ts"]
        return max(0.0, anchor - e["start_ts"] - 1.0)

    def _event_out(e: dict) -> dict:
        e["labels"], e["plates"] = jload(e["labels"], []), jload(e["plates"], [])
        e["rule_name"] = (db.one("SELECT name FROM rules WHERE id=?", (e["rule_id"],)) or {}).get("name") \
            if e["rule_id"] else None
        return e

    @app.get("/api/events")
    def events(camera_id: Optional[str] = None, before: Optional[float] = None, limit: int = Query(100, le=1000)):
        where, args = ["1=1"], []
        if camera_id:
            where.append("camera_id=?")
            args.append(camera_id)
        if before:
            where.append("start_ts<?")
            args.append(before)
        rows = db.all(f"SELECT * FROM events WHERE {' AND '.join(where)} ORDER BY start_ts DESC LIMIT ?",
                      args + [limit])
        return [_event_out(e) for e in rows]

    @app.get("/api/events/{eid}")
    def event(eid: int):
        e = db.one("SELECT * FROM events WHERE id=?", (eid,))
        if not e:
            raise HTTPException(404)
        e = _event_out(e)
        e["motion"] = db.all("SELECT m.* FROM motion_triggers m JOIN event_links l ON l.ref_id=m.id "
                             "AND l.kind='motion' WHERE l.event_id=? ORDER BY m.start_ts", (eid,))
        dets = db.all("SELECT d.id,d.label,d.kind AS animal,d.subtype,d.subtype_conf,d.who,d.confidence,d.first_ts,"
                      "d.last_ts,d.plate,d.snapshot_path,d.zones "
                      "FROM detections d JOIN event_links l ON l.ref_id=d.id AND l.kind='detection' "
                      "WHERE l.event_id=? ORDER BY d.first_ts", (eid,))
        for d in dets:
            d["zones"] = jload(d["zones"], [])
            d["seek"] = max(0.0, e["clip_offset"] + d["first_ts"] - e["start_ts"])
        e["detections"] = dets
        e["play_from"] = e["clip_offset"] + _play_lead_in(e, dets)
        return e

    @app.delete("/api/events/{eid}")
    def event_delete(eid: int):
        e = db.one("SELECT clip_path FROM events WHERE id=?", (eid,))
        if not e:
            raise HTTPException(404)
        db.execute("DELETE FROM events WHERE id=?", (eid,))
        if e["clip_path"] and not db.one("SELECT 1 FROM events WHERE clip_path=?", (e["clip_path"],)):
            if e["clip_path"] not in {w["path"] for w in hub.rec.active().values()}:
                (data / e["clip_path"]).unlink(missing_ok=True)
        return {"ok": True}

    @app.post("/api/events/manual")
    def manual(m: ManualIn):
        cam_or_404(m.camera_id)
        return {"id": hub.manual_event(m.camera_id, m.duration, m.preroll, m.note, "manual")}

    # The AI CPU/GPU header toggle for local programs that need the GPU (e.g. an LLM), with the hook token
    # instead of a session. Registered before /api/hooks/{cam_id} so it isn't taken for a camera id.
    @app.get("/api/hooks/ai-device")
    def hook_ai_device_get():
        det = hub.detector.stats if hub.detector else {}
        return {"device": hub.cfg.detector.device, "active": det.get("device"), "infer_ms": det.get("infer_ms")}

    @app.post("/api/hooks/ai-device")
    def hook_ai_device(device: str = Body(..., embed=True)):
        if device != "cpu" and not device.startswith("cuda"):
            raise HTTPException(422, "device must be 'cpu' or 'cuda:N'")
        if device != hub.cfg.detector.device:
            hub.set_ai_device(device)
            save_config(hub.cfg, config_path)
            configapi.log.info("AI device set to %s by hook", device)
        return {"device": device}

    @app.post("/api/hooks/{cam_id}")
    def hook(cam_id: str, duration: float = 30, preroll: float = 10, body: dict = Body(default={})):
        """External trigger (doorbell, alarm panel, Home Assistant...)."""
        cam_or_404(cam_id)
        return {"id": hub.manual_event(cam_id, duration, preroll, json.dumps(body)[:500] if body else "", "api")}

    # ------------------------------------------------------------ triggers
    @app.get("/api/triggers")
    def triggers(camera_id: Optional[str] = None, limit: int = Query(200, le=2000)):
        q = "SELECT m.*, z.name AS zone_name FROM motion_triggers m LEFT JOIN zones z ON z.id=m.zone_id"
        args: list = []
        if camera_id:
            q += " WHERE m.camera_id=?"
            args.append(camera_id)
        return db.all(q + " ORDER BY m.start_ts DESC LIMIT ?", args + [limit])

    @app.get("/api/detections")
    def detections(camera_id: Optional[str] = None, label: Optional[str] = None,
                   limit: int = Query(200, le=2000)):
        where, args = ["1=1"], []
        if camera_id:
            where.append("camera_id=?")
            args.append(camera_id)
        if label:
            where.append("label=?")
            args.append(label)
        rows = db.all(f"SELECT id,camera_id,label,confidence,first_ts,last_ts,bbox,zones,plate,plate_conf,"
                      f"snapshot_path,event_id FROM detections WHERE {' AND '.join(where)} "
                      f"ORDER BY first_ts DESC LIMIT ?", args + [limit])
        for r in rows:
            r["bbox"], r["zones"] = jload(r["bbox"], []), jload(r["zones"], [])
        return rows

    # -------------------------------------------------------------- search
    @app.post("/api/search")
    async def search(q: str = Body(..., embed=True), cameras: Optional[list[str]] = Body(None, embed=True)):
        """cameras: the cameras ticked in the search bar; omitted = all."""
        res = await asyncio.to_thread(searcher.search, q, cameras)
        ev_cache: dict[int, Optional[dict]] = {}
        for r in res["results"]:
            if r.get("kind") == "detection" and r.get("event_id"):
                eid = r["event_id"]
                if eid not in ev_cache:
                    ev_cache[eid] = db.one("SELECT id,clip_path,clip_offset,start_ts,snapshot_path FROM events WHERE id=?",
                                           (eid,))
                e = ev_cache[eid]
                if e and not r.get("snapshot_path"):     # track cut short (e.g. a restart): use the recording's picture
                    r["snapshot_path"] = e["snapshot_path"]
                if e and e["clip_path"]:
                    r["clip_path"] = e["clip_path"]
                    r["seek"] = max(0.0, e["clip_offset"] + r["first_ts"] - e["start_ts"])
        return res

    # ------------------------------------------------------------ vehicles
    # ------------------------------------------------------------ people (face recognition)
    def people_or_503():
        if not hub.people:
            raise HTTPException(503, "face recognition isn't available")
        return hub.people

    @app.get("/api/people")
    def people_list():
        p = people_or_503()
        out = []
        for pid, name in sorted(p.names().items(), key=lambda kv: kv[1].lower()):
            conf = p.confirmed(pid)
            seen = db.one("SELECT COUNT(*) AS n, MAX(ts) AS last FROM faces WHERE person_id=? AND status IN"
                          " ('auto','confirmed')", (pid,))
            out.append({"id": pid, "name": name, "confirmed": conf[:24], "n_confirmed": len(conf),
                        "n_seen": seen["n"], "last_seen": seen["last"]})
        return {"people": out, "stats": p.stats, "models": bool(p.engine._load())}

    @app.post("/api/people")
    def people_add(name: str = Body(..., embed=True)):
        name = name.strip()
        if not name or len(name) > 40:
            raise HTTPException(422, "name must be 1-40 characters")
        if name.lower() in {n.lower() for n in people_or_503().names().values()}:
            raise HTTPException(409, "that name is already used")
        return {"id": people_or_503().add_person(name)}

    @app.delete("/api/people/{pid}")
    def people_remove(pid: int):
        people_or_503().remove_person(pid)
        return {"ok": True}

    @app.get("/api/people/{pid}/suggestions")
    async def people_suggest(pid: int, limit: int = 60):
        return await asyncio.to_thread(people_or_503().suggestions, pid, min(limit, 200))

    @app.post("/api/faces/{fid}")
    def face_verdict(fid: int, person_id: int = Body(..., embed=True), verdict: str = Body(..., embed=True)):
        """verdict "yes": this face is that person (confirms it); "no": it isn't."""
        p = people_or_503()
        try:
            (p.confirm if verdict == "yes" else p.reject)(fid, person_id)
        except KeyError:
            raise HTTPException(404)
        return {"ok": True}

    @app.post("/api/detections/{det_id}/person")
    async def detection_person(det_id: int, person_id: int = Body(..., embed=True)):
        """'This is <person>' from the event player: enrol this detection's face. person_id 0: 'not them' (undo a
        wrong mark or match)."""
        if person_id == 0:
            return {"changed": await asyncio.to_thread(people_or_503().unmark_detection, det_id)}
        fid = await asyncio.to_thread(people_or_503().enrol_detection, det_id, person_id)
        if not fid:
            raise HTTPException(422, "no usable face found for this detection (too small, turned away or dark)")
        return {"face_id": fid}

    @app.get("/api/vehicles")
    def vehicles(status: str = "all", limit: int = Query(200, le=2000)):
        where, args = ("", []) if status == "all" else ("WHERE status = ?", [status])
        rows = db.all(f"SELECT id, grp, label, plate, sightings, visits, first_ts, last_ts, cameras, snapshot_path,"
                      f" status, flagged_ts, note FROM vehicles {where}"
                      f" ORDER BY status = 'flagged' DESC, visits DESC, last_ts DESC LIMIT ?", args + [limit])
        for r in rows:
            r["cameras"] = jload(r["cameras"], [])
        counts = {r["status"]: r["n"] for r in db.all("SELECT status, COUNT(*) AS n FROM vehicles GROUP BY status")}
        return {"vehicles": rows, "counts": counts, "repeat_visits": hub.cfg.vehicles.repeat_visits}

    @app.get("/api/vehicles/{vid}/sightings")
    def vehicle_sightings(vid: int):
        rows = db.all("SELECT d.id, d.camera_id, d.label, d.confidence, d.first_ts, d.last_ts, d.plate, d.snapshot_path,"
                      " d.event_id, e.clip_path, e.clip_offset, e.start_ts AS ev_start FROM detections d"
                      " LEFT JOIN events e ON e.id = d.event_id WHERE d.vehicle_id = ? ORDER BY d.first_ts DESC", (vid,))
        for r in rows:
            if r["clip_path"]:
                r["seek"] = max(0.0, r["clip_offset"] + r["first_ts"] - r["ev_start"])
        return rows

    @app.put("/api/vehicles/{vid}")
    def vehicle_update(vid: int, status: str = Body(..., pattern="^(new|flagged|known)$"),
                       note: Optional[str] = Body(None)):
        try:
            hub.vehicles.set_status(vid, status, note)
        except KeyError:
            raise HTTPException(404)
        if status == "known":       # a known car (yours, a neighbour's) loses any delivery/police tags it got
            for r in db.all("SELECT id FROM detections WHERE vehicle_id=? AND subtype IS NOT NULL", (vid,)):
                hub.clear_service(r["id"])
        return {"ok": True}

    # --------------------------------------------------------------- misc
    _storage_cache: dict[str, Any] = {}

    @app.get("/api/storage")
    async def storage(refresh: bool = False):
        """Disk used per camera (clips, snapshots) and in total. Walking the folders takes a moment, so cached 60 s."""
        c = _storage_cache.get("v")
        if refresh or not c or time.time() - c["time"] > 60:
            c = _storage_cache["v"] = await asyncio.to_thread(_storage_scan, data, hub, config_path.parent)
        return c

    @app.get("/api/stats")
    def stats():
        return hub.stats() | {"time": time.time()}

    @app.get("/media/{path:path}")
    def media(path: str):
        p = (data / path).resolve()
        if not p.is_relative_to(data) or not p.is_file():
            raise HTTPException(404)
        mt = "video/mp4" if p.suffix == ".mp4" else "image/jpeg" if p.suffix == ".jpg" else None
        return FileResponse(p, media_type=mt)

    configapi.register(app, hub, searcher, config_path)
    auth.register(app, auth_store, WEB)
    class NoCacheStatic(StaticFiles):
        """UI files change with every update; make browsers revalidate (cheap 304 via ETag) instead of
        heuristically caching an old app.js next to a new index.html."""
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    app.mount("/", NoCacheStatic(directory=WEB, html=True), name="web")
    return app
