"""Writes RAM-buffered packets (pre-roll) + live packets (post-roll) to disk, stream-copied.

One clip per camera at a time; overlapping events share it (each event stores its
offset into the clip), so a "motion" rule and a "person" rule firing together don't
write the same footage twice.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import av

from .camera import Camera
from .ringbuffer import Pkt

log = logging.getLogger("theeye.recorder")

FOREVER = float("inf")


class ClipWriter:
    def __init__(self, cam: Camera, path: Path, start_wall: float, max_seconds: float):
        self.cam = cam
        self.path = path
        self.holds = 0
        self.stop_at = FOREVER
        self.max_seconds = max_seconds
        self.clip_start_wall: Optional[float] = None
        self.done = threading.Event()
        self.bytes = 0
        self._q: queue.Queue[Optional[Pkt]] = queue.Queue(maxsize=5000)
        self._gen = cam.ring.generation
        history = cam.ring.snapshot_and_subscribe(start_wall, self._feed)
        for p in history:
            self._q.put(p)
        if history:
            self.clip_start_wall = history[0].wall
        self._t = threading.Thread(target=self._run, name=f"rec-{cam.id}", daemon=True)
        self._t.start()

    def _feed(self, p: Pkt) -> None:
        try:
            self._q.put_nowait(p)
        except queue.Full:
            log.warning("[%s] recorder queue full; dropping packet", self.cam.id)

    def offset_for(self, wall: float) -> float:
        return max(0.0, wall - (self.clip_start_wall or wall))

    def finish_by(self, wall: float) -> None:
        self.stop_at = wall

    def _run(self) -> None:
        tmpl = self.cam.template_stream
        out = None
        try:
            if tmpl is None:
                raise RuntimeError("camera has no main stream")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            out = av.open(str(self.path), "w", format="mp4",
                          options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
            ost = out.add_stream_from_template(tmpl)
            base: Optional[int] = None
            last_dts: Optional[int] = None
            while True:
                try:
                    p = self._q.get(timeout=1.0)
                except queue.Empty:
                    p = None
                now = time.time()
                if self.cam.ring.generation != self._gen:
                    log.info("[%s] stream reconnected; closing clip", self.cam.id)
                    break
                if p is None:
                    if now > self.stop_at:
                        break
                    continue
                if self.clip_start_wall is None:
                    if not p.keyframe:
                        continue
                    self.clip_start_wall = p.wall
                if p.wall > self.stop_at or p.wall - self.clip_start_wall > self.max_seconds:
                    break
                dts = p.dts if p.dts is not None else p.pts
                pts = p.pts if p.pts is not None else dts
                if dts is None:
                    continue
                if base is None:
                    base = dts
                if last_dts is not None and dts - base <= last_dts:
                    continue                      # non-monotonic; mp4 muxer would reject
                last_dts = dts - base
                pkt = av.Packet(p.data)
                pkt.pts, pkt.dts = pts - base, dts - base
                pkt.duration = p.duration
                pkt.time_base = p.time_base
                pkt.is_keyframe = p.keyframe
                pkt.stream = ost
                out.mux(pkt)
                self.bytes += len(p.data)
        except Exception:  # noqa: BLE001
            log.exception("[%s] clip write failed: %s", self.cam.id, self.path)
        finally:
            self.cam.ring.unsubscribe(self._feed)
            if out is not None:
                try:
                    out.close()
                except Exception:  # noqa: BLE001
                    log.exception("closing clip failed")
            self.done.set()
            log.info("[%s] clip closed %s (%.1f MB)", self.cam.id, self.path.name, self.bytes / 1e6)


class RecordingManager:
    def __init__(self, cams: dict[str, Camera], clip_dir: Path, max_clip_seconds: float = 3600):
        self.cams = cams
        self.clip_dir = clip_dir
        self.max_clip_seconds = max_clip_seconds
        self._writers: dict[str, ClipWriter] = {}
        self._lock = threading.Lock()

    def acquire(self, cam_id: str, start_wall: float, name: str) -> tuple[str, float]:
        """Hold the camera's clip open (creating it if needed). Returns (relative path, offset)."""
        with self._lock:
            w = self._writers.get(cam_id)
            if w is None or w.done.is_set() or w.stop_at < time.time():
                day = time.strftime("%Y-%m-%d", time.localtime(start_wall))
                stamp = time.strftime("%H%M%S", time.localtime(start_wall))
                rel = Path("clips") / cam_id / day / f"{stamp}_{name}.mp4"
                w = ClipWriter(self.cams[cam_id], self.clip_dir.parent / rel, start_wall,
                               self.max_clip_seconds)
                w.rel = str(rel)
                self._writers[cam_id] = w
            w.holds += 1
            w.stop_at = FOREVER
            return w.rel, w.offset_for(start_wall)

    def release(self, cam_id: str, rel: str, until_wall: float) -> None:
        with self._lock:
            w = self._writers.get(cam_id)
            if w is None or w.rel != rel:
                return
            w.holds = max(0, w.holds - 1)
            if w.holds == 0:
                w.finish_by(until_wall)

    def active(self) -> dict[str, dict]:
        with self._lock:
            return {c: {"path": w.rel, "holds": w.holds, "mb": round(w.bytes / 1e6, 1)}
                    for c, w in self._writers.items() if not w.done.is_set()}
