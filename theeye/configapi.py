"""Setup page API: read/write config.yaml, probe camera streams, list LLM models, restart."""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import av
import cv2
import httpx
from fastapi import Body, FastAPI, HTTPException
from pydantic import ValidationError

from .camera import stream_opts
from .config import BRANDS, CameraConfig, Config, Credential, redact, resolve_credentials, save_config

log = logging.getLogger("theeye.setup")
MASK = "********"


# ---------------------------------------------------------------- secrets
def mask_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return url
    try:
        u = urlsplit(url)
    except ValueError:
        return url
    if u.password is None:
        return url
    host = u.hostname or ""
    if u.port:
        host += f":{u.port}"
    return urlunsplit(u._replace(netloc=f"{u.username}:{MASK}@{host}"))


def unmask_url(url: Optional[str], old: Optional[str]) -> Optional[str]:
    """If the UI sent back a masked URL, restore the stored password."""
    if not url or MASK not in url or not old:
        return url
    try:
        old_pw = urlsplit(old).password or ""
    except ValueError:
        return url
    return url.replace(MASK, old_pw, 1)


def public_camera(c: CameraConfig) -> dict:
    d = c.model_dump()
    d["has_password"] = bool(c.password) or (c.brand == "custom" and urlsplit(c.main_url).password is not None)
    d["password"] = None
    d["main_url"], d["sub_url"] = mask_url(c.main_url), mask_url(c.sub_url)
    return d


def merge_secrets(cams: list[dict], current: dict[str, CameraConfig]) -> list[dict]:
    """Blank password / masked URL in the form means 'keep what is stored'."""
    out = []
    for c in cams:
        c = {k: v for k, v in c.items() if k != "has_password"}
        old = current.get(c.get("orig_id") or c.get("id"))
        c.pop("orig_id", None)
        if old:
            if not c.get("password"):
                c["password"] = old.password
            c["main_url"] = unmask_url(c.get("main_url"), old.main_url)
            c["sub_url"] = unmask_url(c.get("sub_url"), old.sub_url)
        for k in ("host", "username", "password", "sub_url", "credential"):
            if c.get(k) == "":
                c[k] = None
        if c.get("brand") != "custom":
            c["main_url"], c["sub_url"] = "", None
        out.append(c)
    return out


def public_credential(c: Credential) -> dict:
    return {"name": c.name, "username": c.username, "password": None, "has_password": bool(c.password)}


def merge_credential_secrets(creds: list[dict], current: dict[str, Credential],
                             cams: dict[str, CameraConfig]) -> list[dict]:
    """Blank password keeps the stored one. A login saved from an existing camera's form
    ("from_camera") inherits that camera's stored password, which the browser never sees."""
    out = []
    for c in creds:
        c = dict(c)
        old = current.get(c.pop("orig_name", None) or c.get("name"))
        src = cams.get(c.pop("from_camera", None) or "")
        c.pop("has_password", None)
        if not c.get("password"):
            c["password"] = old.password if old else (src.password if src else None)
        out.append(c)
    return out


def _errors(e: ValidationError) -> list[str]:
    msgs = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msgs.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return msgs


# ------------------------------------------------------------------ probe
def probe_stream(url: str, seconds: float = 4.0) -> dict[str, Any]:
    """Connect, sample a few seconds of packets: codec, resolution, fps, bitrate, keyframe interval, snapshot."""
    t0 = time.time()
    c = av.open(url, options=stream_opts(url), timeout=(8.0, 8.0))
    try:
        if not c.streams.video:
            raise RuntimeError("no video stream found")
        vs = c.streams.video[0]
        audio = c.streams.audio[0].codec_context.name if c.streams.audio else None
        connect_ms = int((time.time() - t0) * 1000)
        nbytes = npk = 0
        kf: list[float] = []
        first = last = None
        img = None
        decoding = False
        t1 = time.time()
        for pkt in c.demux(vs):
            if pkt.size == 0:
                continue
            npk += 1
            nbytes += pkt.size
            if pkt.pts is not None and pkt.time_base:
                ts = float(pkt.pts * pkt.time_base)
                first = ts if first is None else first
                last = ts
                if pkt.is_keyframe:
                    kf.append(ts)
            if img is None and (decoding or pkt.is_keyframe):
                decoding = True
                for fr in pkt.decode():
                    img = fr.to_ndarray(format="bgr24")
                    break
            el = time.time() - t1
            if (el > seconds and len(kf) >= 2 and img is not None) or el > seconds * 3:
                break
        dur = (last - first) if first is not None and last is not None and last > first else time.time() - t1
        gaps = [b - a for a, b in zip(kf, kf[1:]) if b > a]
        fps = float(vs.average_rate) if vs.average_rate else (npk / dur if dur else None)
        snap = None
        if img is not None:
            h, w = img.shape[:2]
            if w > 640:
                img = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            snap = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None
        return {
            "ok": True, "codec": vs.codec_context.name, "width": vs.codec_context.width,
            "height": vs.codec_context.height, "fps": round(fps, 1) if fps else None, "audio": audio,
            "bitrate_kbps": round(nbytes * 8 / dur / 1000) if dur > 0 else None,
            "gop_seconds": round(statistics.median(gaps), 2) if gaps else None,
            "connect_ms": connect_ms, "snapshot": snap,
        }
    finally:
        c.close()


def total_ram() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return 0


# ----------------------------------------------------------------- routes
def register(app: FastAPI, hub, searcher, config_path: Path) -> None:
    lock = threading.Lock()

    def current_map() -> dict[str, CameraConfig]:
        return {c.id: c for c in hub.cfg.cameras}

    @app.get("/api/config")
    def get_config():
        d = hub.cfg.model_dump()
        d["cameras"] = [public_camera(c) for c in hub.cfg.cameras]
        d["credentials"] = [public_credential(c) for c in hub.cfg.credentials]
        return {"config": d, "path": str(config_path), "exists": config_path.exists(),
                "brands": {k: {"label": v["label"], "port": v["port"], "scheme": v.get("scheme", "rtsp"),
                               "has_sub": bool(v["sub"]) or k == "custom", "note": v.get("note", "")}
                           for k, v in BRANDS.items()},
                "system": {"ram_bytes": total_ram()}}

    @app.put("/api/config")
    def put_config(body: dict = Body(...)):
        with lock:
            body = dict(body)
            body["credentials"] = merge_credential_secrets(
                body.get("credentials") or [], {c.name: c for c in hub.cfg.credentials}, current_map())
            body["cameras"] = merge_secrets(body.get("cameras") or [], current_map())
            ids = [c.get("id") for c in body["cameras"]]
            if len(ids) != len(set(ids)):
                raise HTTPException(422, {"errors": ["camera ids must be unique"]})
            try:
                new = Config.model_validate(body)
            except ValidationError as e:
                raise HTTPException(422, {"errors": _errors(e)})
            save_config(new, config_path)
            restart = hub.apply_config(new)
            searcher.cfg = new.llm
            searcher.cameras = {c.id: c.cfg.name or c.id for c in hub.cams.values()}
        log.info("config saved to %s (restart needed for: %s)", config_path, restart or "nothing")
        return {"saved": True, "restart_required": restart}

    @app.put("/api/cameras/order")
    def reorder(ids: list[str] = Body(..., embed=True)):
        """Reorder cameras (e.g. dragged in Live view). Only the listed cameras move, taking the slots they
        already occupy, so cameras the client didn't show (disabled ones) keep their place. No restarts."""
        with lock:
            cams = list(hub.cfg.cameras)
            known = {c.id: c for c in cams}
            ids = [i for i in dict.fromkeys(ids) if i in known]
            slots = [n for n, c in enumerate(cams) if c.id in set(ids)]
            for n, cid in zip(slots, ids):
                cams[n] = known[cid]
            new = hub.cfg.model_copy(update={"cameras": cams})
            save_config(new, config_path)
            hub.cfg = new
        return {"order": [c.id for c in cams]}

    @app.post("/api/config/probe")
    async def probe(camera: dict = Body(...), stream: str = Body("main"), credentials: list = Body([])):
        cams = merge_secrets([camera], current_map())
        creds = merge_credential_secrets(credentials, {c.name: c for c in hub.cfg.credentials}, current_map())
        try:
            cam = resolve_credentials(cams, creds)[0]
            cfg = CameraConfig.model_validate(cam | {"id": cam.get("id") or "probe"})
        except ValueError as e:      # includes ValidationError
            return {"ok": False, "error": "; ".join(_errors(e)) if isinstance(e, ValidationError) else str(e)}
        url = cfg.sub_url if stream == "sub" else cfg.main_url
        if not url:
            return {"ok": False, "error": "no URL for this stream"}
        try:
            res = await asyncio.wait_for(asyncio.to_thread(probe_stream, url), timeout=30)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timed out", "url": mask_url(url)}
        except Exception as e:  # noqa: BLE001 - surface camera/ffmpeg errors to the form
            msg = redact(str(e))
            if cfg.password:
                msg = msg.replace(cfg.password, MASK)
            return {"ok": False, "error": msg, "url": mask_url(url)}
        res["url"] = mask_url(url)
        return res

    @app.get("/api/config/llm-models")
    async def llm_models(url: Optional[str] = None):
        base = (url or hub.cfg.llm.url).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5) as cl:
                r = await cl.get(f"{base}/api/tags")
                r.raise_for_status()
            models = r.json().get("models", [])
            return {"ok": True, "models": [{"name": m["name"], "size": m.get("size", 0)} for m in models]}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}", "models": []}

    @app.post("/api/restart")
    def restart():
        def go():
            time.sleep(0.5)
            log.info("restarting…")
            try:
                hub.stop()
                time.sleep(1.5)            # let clip writers flush
            finally:
                os.execv(sys.executable, [sys.executable, "-m", "theeye", *sys.argv[1:]])
        threading.Thread(target=go, daemon=True).start()
        return {"restarting": True}
