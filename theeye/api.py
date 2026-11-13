"""HTTP API + static web UI."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import configapi
from .db import jload
from .hub import Hub
from .search import Searcher

WEB = Path(__file__).parent / "web"


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

    def cam_or_404(cam_id: str):
        c = hub.cams.get(cam_id)
        if not c:
            raise HTTPException(404, "camera not found")
        return c

    # ------------------------------------------------------------- cameras
    @app.get("/api/cameras")
    def cameras():
        return [{"id": c.id, "name": c.cfg.name or c.id, "status": c.status, "ring": c.ring.stats,
                 "ring_seconds": c.cfg.ring_seconds} for c in hub.cams.values()]

    @app.get("/api/cameras/{cam_id}/snapshot.jpg")
    def snapshot(cam_id: str):
        c = cam_or_404(cam_id)
        if not c.latest_jpeg:
            raise HTTPException(503, "no frame yet")
        return Response(c.latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

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
        return rows

    def _zone_params(z: ZoneIn):
        if len(z.points) < 3:
            raise HTTPException(400, "zone needs at least 3 points")
        pts = [[min(1.0, max(0.0, x)), min(1.0, max(0.0, y))] for x, y in z.points]
        return (z.camera_id, z.name, json.dumps(pts), z.kind, z.sensitivity, z.min_area, z.min_frames,
                z.color, int(z.enabled))

    @app.post("/api/zones")
    def zone_create(z: ZoneIn):
        cam_or_404(z.camera_id)
        zid = db.insert("INSERT INTO zones(camera_id,name,points,kind,sensitivity,min_area,min_frames,color,enabled)"
                        " VALUES(?,?,?,?,?,?,?,?,?)", _zone_params(z))
        hub.reload_zones()
        return {"id": zid}

    @app.put("/api/zones/{zid}")
    def zone_update(zid: int, z: ZoneIn):
        db.execute("UPDATE zones SET camera_id=?,name=?,points=?,kind=?,sensitivity=?,min_area=?,min_frames=?,"
                   "color=?,enabled=? WHERE id=?", _zone_params(z) + (zid,))
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
        dets = db.all("SELECT d.id,d.label,d.confidence,d.first_ts,d.last_ts,d.plate,d.snapshot_path,d.zones "
                      "FROM detections d JOIN event_links l ON l.ref_id=d.id AND l.kind='detection' "
                      "WHERE l.event_id=? ORDER BY d.first_ts", (eid,))
        for d in dets:
            d["zones"] = jload(d["zones"], [])
            d["seek"] = max(0.0, e["clip_offset"] + d["first_ts"] - e["start_ts"])
        e["detections"] = dets
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
    async def search(q: str = Body(..., embed=True)):
        res = await asyncio.to_thread(searcher.search, q)
        ev_cache: dict[int, Optional[dict]] = {}
        for r in res["results"]:
            if r.get("kind") == "detection" and r.get("event_id"):
                eid = r["event_id"]
                if eid not in ev_cache:
                    ev_cache[eid] = db.one("SELECT id,clip_path,clip_offset,start_ts FROM events WHERE id=?", (eid,))
                e = ev_cache[eid]
                if e and e["clip_path"]:
                    r["clip_path"] = e["clip_path"]
                    r["seek"] = max(0.0, e["clip_offset"] + r["first_ts"] - e["start_ts"])
        return res

    # --------------------------------------------------------------- misc
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
    class NoCacheStatic(StaticFiles):
        """UI files change with every update; make browsers revalidate (cheap 304 via ETag) instead of
        heuristically caching an old app.js next to a new index.html."""
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    app.mount("/", NoCacheStatic(directory=WEB, html=True), name="web")
    return app
