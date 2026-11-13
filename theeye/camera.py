"""Per-camera ingest.

main stream  -> demux only (no decode) -> RingBuffer of compressed packets
sub stream   -> decode -> sampled BGR frames -> analysis callbacks (motion/detector/live view)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import av
import cv2
import numpy as np

from .config import CameraConfig, redact
from .ringbuffer import Pkt, RingBuffer

log = logging.getLogger("theeye.camera")

RTSP_OPTS = {
    "rtsp_transport": "tcp",
    "stimeout": "5000000",       # socket timeout (us) – older ffmpeg name
    "timeout": "5000000",        # newer ffmpeg name
    "fflags": "nobuffer+genpts",
    "max_delay": "500000",
}
ANALYSIS_WIDTH = 640


def stream_opts(url: str) -> dict:
    if url.startswith("rtsps://"):
        # Cameras (e.g. Wyze) use self-signed certs, and newer ffmpeg verifies by default, failing with
        # "Peer certificate failed verification". Traffic stays encrypted; only the identity check is skipped.
        return {**RTSP_OPTS, "tls_verify": "0"}
    return dict(RTSP_OPTS) if url.startswith("rtsp://") else {}


AUTH_RETRY_S = 300.0     # cameras (Wyze included) lock out clients that keep failing to log in
MAX_RETRY_S = 60.0


def _is_auth_error(e: Exception) -> bool:
    return isinstance(e, getattr(av, "HTTPUnauthorizedError", ())) or "401" in str(e) or "nauthorized" in str(e)


def _next_backoff(e: Exception, backoff: float) -> float:
    return AUTH_RETRY_S if _is_auth_error(e) else min(max(backoff, 1.0) * 2, MAX_RETRY_S)


def _open(url: str) -> av.container.InputContainer:
    return av.open(url, options=stream_opts(url), timeout=(10.0, 10.0))


class Camera:
    def __init__(self, cfg: CameraConfig, ring_bytes: int):
        self.cfg = cfg
        self.id = cfg.id
        self.ring = RingBuffer(cfg.ring_seconds, ring_bytes)
        self.template_stream: Optional[av.video.stream.VideoStream] = None  # for stream-copy muxing
        self.main_size: tuple[int, int] = (0, 0)
        self.latest_frame: Optional[np.ndarray] = None     # analysis-res BGR
        self.latest_frame_ts: float = 0.0
        self.latest_jpeg: Optional[bytes] = None
        self.frame_cond = threading.Condition()
        self._frame_cbs: list[Callable[[str, np.ndarray, float], None]] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.status = {"main": "init", "sub": "init", "main_err": None, "sub_err": None,
                       "main_codec": None, "sub_res": None}
        self._main_container = None
        self._hd_lock = threading.Lock()
        self._hd_cache: Optional[tuple[Optional[np.ndarray], float]] = None

    def on_frame(self, cb: Callable[[str, np.ndarray, float], None]) -> None:
        self._frame_cbs.append(cb)

    def start(self) -> None:
        t = threading.Thread(target=self._main_loop, name=f"main-{self.id}", daemon=True)
        self._threads.append(t)
        t.start()
        if self.cfg.sub_url:
            t2 = threading.Thread(target=self._sub_loop, name=f"sub-{self.id}", daemon=True)
            self._threads.append(t2)
            t2.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ main
    def _main_loop(self) -> None:
        backoff = 1.0
        analyse_here = not self.cfg.sub_url
        while not self._stop.is_set():
            c = None
            try:
                c = _open(self.cfg.main_url)
                vs = c.streams.video[0]
                self.ring.reset()
                self.template_stream = vs
                self._main_container = c
                self.main_size = (vs.codec_context.width, vs.codec_context.height)
                self.status.update(main="live", main_err=None,
                                   main_codec=f"{vs.codec_context.name} {self.main_size[0]}x{self.main_size[1]}")
                log.info("[%s] main stream up: %s", self.id, self.status["main_codec"])
                backoff = 1.0
                sampler = _Sampler(self.cfg.analyze_fps)
                for pkt in c.demux(vs):
                    if self._stop.is_set():
                        break
                    if pkt.size == 0:
                        continue
                    now = time.time()
                    self.ring.push(Pkt(bytes(pkt), pkt.pts, pkt.dts, pkt.duration or 0,
                                       pkt.time_base, bool(pkt.is_keyframe), now))
                    if analyse_here:
                        for fr in pkt.decode():
                            if sampler.take(now):
                                self._emit(fr, now)
                backoff = 1.0                # stream ended normally: reconnect promptly
            except Exception as e:  # noqa: BLE001 - camera errors are expected; keep retrying
                backoff = _next_backoff(e, backoff)
                err = redact(str(e))
                self.status.update(main="error", main_err=err)
                log.warning("[%s] main stream error: %s (retry in %.0fs)", self.id, err, backoff)
            finally:
                if c is not None:            # always release the camera-side session
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001
                        pass
            self._stop.wait(backoff)

    # ------------------------------------------------------------------- sub
    def _sub_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            c = None
            try:
                c = _open(self.cfg.sub_url)
                vs = c.streams.video[0]
                vs.thread_type = "AUTO"
                self.status.update(sub="live", sub_err=None,
                                   sub_res=f"{vs.codec_context.width}x{vs.codec_context.height}")
                backoff = 1.0
                sampler = _Sampler(self.cfg.analyze_fps)
                for fr in c.decode(vs):
                    if self._stop.is_set():
                        break
                    now = time.time()
                    if sampler.take(now):
                        self._emit(fr, now)
                backoff = 1.0
            except Exception as e:  # noqa: BLE001
                backoff = _next_backoff(e, backoff)
                err = redact(str(e))
                self.status.update(sub="error", sub_err=err)
                log.warning("[%s] sub stream error: %s (retry in %.0fs)", self.id, err, backoff)
            finally:
                if c is not None:
                    try:
                        c.close()
                    except Exception:  # noqa: BLE001
                        pass
            self._stop.wait(backoff)

    def _emit(self, fr: av.VideoFrame, now: float) -> None:
        if self._stop.is_set():          # camera removed/replaced: old thread may linger until next packet
            return
        w = fr.width
        if w > ANALYSIS_WIDTH:
            h = int(fr.height * ANALYSIS_WIDTH / w) // 2 * 2
            img = fr.to_ndarray(format="bgr24", width=ANALYSIS_WIDTH, height=h)
        else:
            img = fr.to_ndarray(format="bgr24")
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self.frame_cond:
            self.latest_frame = img
            self.latest_frame_ts = now
            if ok:
                self.latest_jpeg = jpg.tobytes()
            self.frame_cond.notify_all()
        for cb in self._frame_cbs:
            try:
                cb(self.id, img, now)
            except Exception:  # noqa: BLE001
                log.exception("[%s] frame callback failed", self.id)

    # --------------------------------------------------------------- helpers
    def decode_latest_frame(self, max_age: float = 0.4) -> Optional[np.ndarray]:
        """Full-resolution main-stream frame (decodes the latest GOP up to now).

        Used for licence plates, which are unreadable on the sub stream. Cached
        briefly so several vehicles in one frame share a decode.
        """
        now = time.time()
        with self._hd_lock:
            if self._hd_cache is not None and now - self._hd_cache[1] < max_age:
                return self._hd_cache[0]
            gop = self.ring.latest_gop()
            ts = self.template_stream
            if not gop or ts is None:
                return None
            try:
                cc = av.CodecContext.create(ts.codec_context.name, "r")
                if ts.codec_context.extradata:
                    cc.extradata = ts.codec_context.extradata
                cc.thread_type = "AUTO"
                last = None
                for p in gop:
                    for fr in cc.decode(av.Packet(p.data)):
                        last = fr
                for fr in cc.decode(None):
                    last = fr
                img = last.to_ndarray(format="bgr24") if last is not None else None
                self._hd_cache = (img, now)
                return img
            except Exception:  # noqa: BLE001
                log.debug("[%s] main frame decode failed", self.id, exc_info=True)
                return None

    def wait_jpeg(self, last_ts: float, timeout: float = 2.0) -> tuple[Optional[bytes], float]:
        with self.frame_cond:
            self.frame_cond.wait_for(lambda: self.latest_frame_ts > last_ts, timeout=timeout)
            return self.latest_jpeg, self.latest_frame_ts


class _Sampler:
    def __init__(self, fps: float):
        self.interval = 1.0 / max(fps, 0.1)
        self.next = 0.0

    def take(self, now: float) -> bool:
        if now >= self.next:
            self.next = now + self.interval
            return True
        return False
