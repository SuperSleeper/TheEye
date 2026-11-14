"""In-RAM loop of *compressed* packets from a camera's main stream.

Storing encoded packets (not decoded frames) is what makes a long loop cheap:
a 4MP H.264/H.265 stream at ~6 Mbit/s is ~45 MB/min. The buffer is trimmed on
GOP boundaries so any read can always start on a keyframe.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional


@dataclass(slots=True)
class Pkt:
    data: bytes
    pts: Optional[int]
    dts: Optional[int]
    duration: int
    time_base: Fraction
    keyframe: bool
    wall: float          # wall-clock time packet arrived (epoch seconds)
    audio: bool = False  # audio packet interleaved with the video (never a keyframe; skipped by video decoders)


class RingBuffer:
    def __init__(self, seconds: float, max_bytes: int):
        self.seconds = seconds
        self.max_bytes = max_bytes
        self._q: deque[Pkt] = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self._subs: list[Callable[[Pkt], None]] = []
        self.generation = 0          # bumps when stream (re)connects; codec params may change

    # ---- writer side -------------------------------------------------------
    def push(self, p: Pkt) -> None:
        with self._lock:
            # Never start the buffer on a non-keyframe.
            if not self._q and not p.keyframe:
                return
            self._q.append(p)
            self._bytes += len(p.data)
            self._trim(p.wall)
            subs = list(self._subs)
        for cb in subs:
            cb(p)

    def _trim(self, now: float) -> None:
        # Drop whole GOPs from the front while older than the window or over budget.
        while len(self._q) > 1:
            too_old = now - self._q[0].wall > self.seconds
            too_big = self._bytes > self.max_bytes
            if not (too_old or too_big):
                break
            # find next keyframe after the head
            nxt = None
            for i in range(1, len(self._q)):
                if self._q[i].keyframe:
                    nxt = i
                    break
            if nxt is None:
                break
            for _ in range(nxt):
                self._bytes -= len(self._q.popleft().data)

    def reset(self) -> None:
        with self._lock:
            self._q.clear()
            self._bytes = 0
            self.generation += 1

    # ---- reader side -------------------------------------------------------
    def snapshot_from(self, since_wall: float) -> list[Pkt]:
        """Packets starting at the last keyframe at/before since_wall, through now."""
        with self._lock:
            pk = list(self._q)
        start = 0
        for i, p in enumerate(pk):
            if p.keyframe and p.wall <= since_wall:
                start = i
            elif p.wall > since_wall:
                break
        return pk[start:]

    def gop_until(self, wall: float) -> list[Pkt]:
        """Packets from the last keyframe at/before `wall` through the packet arriving at `wall`."""
        with self._lock:
            pk = list(self._q)
        end = next((i for i, p in enumerate(pk) if p.wall > wall), len(pk))
        for i in range(end - 1, -1, -1):
            if pk[i].keyframe:
                return pk[i:max(end, i + 1)]
        return []

    def latest_gop(self) -> list[Pkt]:
        with self._lock:
            pk = list(self._q)
        for i in range(len(pk) - 1, -1, -1):
            if pk[i].keyframe:
                return pk[i:]
        return []

    def snapshot_and_subscribe(self, since_wall: float, cb: Callable[[Pkt], None]) -> list[Pkt]:
        """Atomically grab history from since_wall and subscribe to new packets (no gap/dupes)."""
        with self._lock:
            pk = list(self._q)
            self._subs.append(cb)
        start = 0
        for i, p in enumerate(pk):
            if p.keyframe and p.wall <= since_wall:
                start = i
            elif p.wall > since_wall:
                break
        return pk[start:]

    def subscribe(self, cb: Callable[[Pkt], None]) -> None:
        with self._lock:
            self._subs.append(cb)

    def unsubscribe(self, cb: Callable[[Pkt], None]) -> None:
        with self._lock:
            if cb in self._subs:
                self._subs.remove(cb)

    @property
    def stats(self) -> dict:
        with self._lock:
            span = (self._q[-1].wall - self._q[0].wall) if len(self._q) > 1 else 0.0
            return {"packets": len(self._q), "bytes": self._bytes, "seconds": round(span, 1)}
