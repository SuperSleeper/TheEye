"""Writes RAM-buffered packets (pre-roll) + live packets (post-roll) to disk, stream-copied.

One clip per camera at a time; overlapping events share it (each event stores its
offset into the clip), so a "motion" rule and a "person" rule firing together don't
write the same footage twice.
"""
from __future__ import annotations

import io
import logging
import queue
import threading
import time
from fractions import Fraction
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
        self._ready = threading.Event()        # live packets wait until history is queued (keeps order)
        history = cam.ring.snapshot_and_subscribe(start_wall, self._feed)
        for p in history:
            self._q.put(p)
        self._ready.set()
        if history:
            self.clip_start_wall = history[0].wall
        self._t = threading.Thread(target=self._run, name=f"rec-{cam.id}", daemon=True)
        self._t.start()

    def _feed(self, p: Pkt) -> None:
        self._ready.wait(2.0)
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
        audio = None
        try:
            if tmpl is None:
                raise RuntimeError("camera has no main stream")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            out = av.open(str(self.path), "w", format="mp4",
                          options={"movflags": "frag_keyframe+empty_moov+default_base_moof"})
            ost = out.add_stream_from_template(tmpl)
            audio = _AudioTrack.create(out, self.cam.audio_template, self.cam.id)
            base: Optional[int] = None
            base_s: Optional[float] = None      # first video packet's time, in seconds (audio aligns to it)
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
                if p.audio:
                    if audio is not None and base_s is not None and p.wall <= self.stop_at:
                        self.bytes += audio.write(out, p, base_s)
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
                    base_s = float(dts * p.time_base)
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
            if out is not None and audio is not None:
                try:
                    audio.flush(out)
                except Exception:  # noqa: BLE001
                    log.debug("[%s] audio flush failed", self.cam.id, exc_info=True)
            if out is not None:
                try:
                    out.close()
                except Exception:  # noqa: BLE001
                    log.exception("closing clip failed")
            self.done.set()
            log.info("[%s] clip closed %s (%.1f MB)", self.cam.id, self.path.name, self.bytes / 1e6)


class _AudioTrack:
    """Audio in clips. AAC is stream-copied; anything else (Wyze Cam v3 sends G.711 A-law, which MP4 can't
    hold) is decoded and re-encoded to AAC. Timestamps are aligned to the clip's first video packet using
    the camera's own stream clock (RTSP keeps audio and video on a common timeline)."""

    def __init__(self, ost, dec=None, enc_resampler=None):
        self.ost, self.dec, self.resampler = ost, dec, enc_resampler
        self.last_dts: Optional[int] = None
        self.next_pts: Optional[int] = None   # transcode path: running sample position

    @classmethod
    def create(cls, out, tmpl, cam_id: str) -> Optional["_AudioTrack"]:
        if tmpl is None:
            return None
        try:
            name = tmpl.codec_context.name
            if name == "aac":
                return cls(out.add_stream_from_template(tmpl))
            rate = tmpl.codec_context.sample_rate or 8000
            dec = av.CodecContext.create(name, "r")
            dec.sample_rate = rate
            dec.layout = tmpl.codec_context.layout.name if tmpl.codec_context.layout else "mono"
            ost = out.add_stream("aac", rate=rate)
            ost.codec_context.layout = "mono"
            ost.codec_context.bit_rate = 32000
            ost.time_base = Fraction(1, rate)
            res = av.AudioResampler(format="fltp", layout="mono", rate=rate,
                                    frame_size=ost.codec_context.frame_size or 1024)
            return cls(ost, dec, res)
        except Exception:  # noqa: BLE001
            log.warning("[%s] audio track unavailable; recording video only", cam_id, exc_info=True)
            return None

    def write(self, out, p: Pkt, base_s: float) -> int:
        t = (p.pts if p.pts is not None else p.dts)
        if t is None:
            return 0
        rel_s = float(t * p.time_base) - base_s
        if rel_s < 0:
            return 0                       # audio from before the clip's first video frame
        if self.dec is None:               # AAC: copy
            # Every AAC frame is 1024 samples. Cameras (Wyze Cam v4) often send no duration and jittery
            # timestamps (40/80 ms apart for 64 ms frames); in fragmented MP4 the overlaps/gaps make the
            # browser discard audio, so lay frames end to end, re-syncing only on a real gap (packet loss).
            sr = self.ost.codec_context.sample_rate or 0
            dur = p.duration or (int(round(1024 / sr / p.time_base)) if sr else 0)
            ts = int(round(rel_s / p.time_base))
            if self.next_pts is not None and dur and abs(ts - self.next_pts) * p.time_base < 0.25:
                ts = self.next_pts
            if self.last_dts is not None and ts <= self.last_dts:
                return 0
            self.last_dts = ts
            self.next_pts = ts + dur
            pkt = av.Packet(p.data)
            pkt.pts = pkt.dts = ts
            pkt.duration = dur
            pkt.time_base = p.time_base
            pkt.stream = self.ost
            out.mux(pkt)
            return len(p.data)
        n = 0
        rate = self.ost.codec_context.sample_rate
        for fr in self.dec.decode(av.Packet(p.data)):
            if self.next_pts is None:
                self.next_pts = int(round(rel_s * rate))
            for rf in self.resampler.resample(fr):
                rf.pts, rf.time_base = self.next_pts, Fraction(1, rate)
                self.next_pts += rf.samples
                for ep in self.ost.encode(rf):
                    out.mux(ep)
                    n += ep.size
        return n

    def flush(self, out) -> None:
        if self.dec is None:
            return
        for rf in self.resampler.resample(None):
            rf.pts, rf.time_base = self.next_pts or 0, Fraction(1, self.ost.codec_context.sample_rate)
            for ep in self.ost.encode(rf):
                out.mux(ep)
        for ep in self.ost.encode(None):
            out.mux(ep)


class _Sink(io.RawIOBase):
    """File-like target for the MP4 muxer that just collects what it writes."""

    def __init__(self):
        super().__init__()
        self._chunks: list[bytes] = []

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self._chunks.append(bytes(b))
        return len(b)

    def take(self) -> bytes:
        out, self._chunks = b"".join(self._chunks), []
        return out


def mse_mime(video_stream, has_audio: bool, audio_profile: Optional[str] = None) -> str:
    """MIME type with RFC 6381 codec strings, as MediaSource.addSourceBuffer() needs."""
    cc = video_stream.codec_context
    codecs = []
    if cc.name == "h264":
        ex = bytes(cc.extradata or b"")
        sps = None
        if ex[:1] == b"\x01" and len(ex) >= 4:                 # avcC: version, profile, compat, level
            sps = ex[1:4]
        else:                                                   # Annex B: find the SPS NAL (type 7)
            for start in (b"\x00\x00\x00\x01", b"\x00\x00\x01"):
                i = ex.find(start)
                while i >= 0:
                    j = i + len(start)
                    if j + 3 < len(ex) and ex[j] & 0x1F == 7:
                        sps = ex[j + 1:j + 4]
                        break
                    i = ex.find(start, j)
                if sps:
                    break
        codecs.append("avc1." + (sps.hex() if sps and len(sps) == 3 else "42e01f"))
    elif cc.name == "hevc":
        codecs.append("hvc1.1.6.L120.90")
    if has_audio:
        codecs.append("mp4a.40.5" if (audio_profile or "").upper().startswith("HE") else "mp4a.40.2")
    return f'video/mp4; codecs="{",".join(codecs)}"'


class LiveMP4:
    """Endless fragmented MP4 of a camera's full-resolution main stream (plus audio), for the full-window
    live view. Packets come from the RAM loop, so it starts instantly at the latest keyframe, needs no
    extra camera connection and no re-encoding (the browser decodes the camera's own H.264)."""

    def __init__(self, cam: Camera, with_audio: bool = True):
        self.cam = cam
        self._q: queue.Queue[Pkt] = queue.Queue(maxsize=3000)
        self._sink = _Sink()
        self._gen = cam.ring.generation
        self.finished = False
        # 200 ms fragments: low latency without frag_every_frame, which writes single-sample audio
        # fragments with a 0 duration (the browser then buffers almost no audio and playback stutters).
        self.out = av.open(self._sink, "w", format="mp4",
                           options={"movflags": "frag_keyframe+empty_moov+default_base_moof",
                                    "frag_duration": "200000"})
        self.ost = self.out.add_stream_from_template(cam.template_stream)
        self.audio = _AudioTrack.create(self.out, cam.audio_template, cam.id) if with_audio else None
        at = cam.audio_template
        self.mime = mse_mime(cam.template_stream, self.audio is not None,
                             at.codec_context.profile if (at is not None and at.codec_context.name == "aac") else None)
        self.base: Optional[int] = None
        self.base_s: Optional[float] = None
        self.last_dts: Optional[int] = None
        self._ready = threading.Event()
        for p in cam.ring.snapshot_and_subscribe(time.time(), self._feed):   # latest GOP onward
            self._put(p)
        self._ready.set()

    def _feed(self, p: Pkt) -> None:
        self._ready.wait(2.0)
        self._put(p)

    def _put(self, p: Pkt) -> None:
        try:
            self._q.put_nowait(p)
        except queue.Full:            # viewer can't keep up: end the stream (the page reconnects)
            self.finished = True

    def read(self, timeout: float = 1.0) -> Optional[bytes]:
        """Mux what has arrived; returns new MP4 bytes (b"" if nothing yet) or None when the stream is over."""
        if self.finished or self.cam.ring.generation != self._gen:
            return None
        try:
            p = self._q.get(timeout=timeout)
        except queue.Empty:
            return b""
        while p is not None:
            self._mux(p)
            try:
                p = self._q.get_nowait()
            except queue.Empty:
                p = None
        return self._sink.take()

    def _mux(self, p: Pkt) -> None:
        if p.audio:
            if self.audio is not None and self.base_s is not None:
                self.audio.write(self.out, p, self.base_s)
            return
        dts = p.dts if p.dts is not None else p.pts
        pts = p.pts if p.pts is not None else dts
        if dts is None or (self.base is None and not p.keyframe):
            return
        if self.base is None:
            self.base, self.base_s = dts, float(dts * p.time_base)
        if self.last_dts is not None and dts - self.base <= self.last_dts:
            return
        self.last_dts = dts - self.base
        pkt = av.Packet(p.data)
        pkt.pts, pkt.dts = pts - self.base, dts - self.base
        pkt.duration, pkt.time_base, pkt.is_keyframe, pkt.stream = p.duration, p.time_base, p.keyframe, self.ost
        self.out.mux(pkt)

    def close(self) -> None:
        self.finished = True
        self.cam.ring.unsubscribe(self._feed)
        try:
            self.out.close()
        except Exception:  # noqa: BLE001
            pass


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
