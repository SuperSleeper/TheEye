"""Zone-based motion detection on analysis frames.

Produces *motion triggers* (start/end per zone). Triggers are only signals; whether
they create an *event* is decided by the rules engine.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

WORK_WIDTH = 320
FULL_FRAME = 0           # zone id used when a camera has no motion zones drawn


@dataclass
class Zone:
    id: int
    camera_id: str
    name: str
    points: list[list[float]]
    kind: str = "motion"            # motion | detect | exclude
    sensitivity: float = 0.5
    min_area: float = 0.02
    min_frames: int = 2
    enabled: bool = True

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon on normalized coordinates."""
        pts = np.asarray(self.points, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0


@dataclass
class _ZoneState:
    hits: int = 0
    misses: int = 0
    active: bool = False
    peak: float = 0.0


@dataclass
class _CamState:
    bg: Optional[np.ndarray] = None
    shape: tuple[int, int] = (0, 0)
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    exclude: Optional[np.ndarray] = None
    zstate: dict[int, _ZoneState] = field(default_factory=dict)
    last_ts: float = 0.0


class MotionEngine:
    def __init__(self,
                 on_start: Callable[[str, int, float, float], None],
                 on_end: Callable[[str, int, float, float], None],
                 release_seconds: float = 2.0):
        self.on_start = on_start
        self.on_end = on_end
        self.release_seconds = release_seconds
        self._zones: dict[str, list[Zone]] = {}
        self._cams: dict[str, _CamState] = {}
        self._lock = threading.Lock()

    def set_zones(self, zones: list[Zone]) -> None:
        by_cam: dict[str, list[Zone]] = {}
        for z in zones:
            by_cam.setdefault(z.camera_id, []).append(z)
        with self._lock:
            self._zones = by_cam
            for st in self._cams.values():   # force mask rebuild; keep background
                st.masks.clear()
                st.exclude = None
                st.shape = (0, 0)

    def forget(self, cam: str) -> None:
        """Drop background model/state (camera removed or reconfigured)."""
        with self._lock:
            self._cams.pop(cam, None)

    def zones_for(self, cam: str) -> list[Zone]:
        with self._lock:
            return list(self._zones.get(cam, []))

    def active_zones(self, cam: str) -> set[int]:
        st = self._cams.get(cam)
        if not st:
            return set()
        return {zid for zid, s in st.zstate.items() if s.active}

    # -----------------------------------------------------------------------
    def process(self, cam: str, img: np.ndarray, ts: float) -> None:
        h0, w0 = img.shape[:2]
        h = int(h0 * WORK_WIDTH / w0)
        gray = cv2.cvtColor(cv2.resize(img, (WORK_WIDTH, h), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        with self._lock:
            zones = [z for z in self._zones.get(cam, []) if z.enabled]
            st = self._cams.setdefault(cam, _CamState())
            stale: list[tuple[int, _ZoneState]] = []
            if st.shape != gray.shape:
                reshape = st.bg is not None and st.bg.shape != gray.shape
                self._build_masks(st, zones, gray.shape)
                if reshape:
                    st.bg = None
                for zid in list(st.zstate):
                    if zid not in st.masks:
                        zs = st.zstate.pop(zid)
                        if zs.active:
                            stale.append((zid, zs))
        for zid, zs in stale:           # zone deleted/changed while active: close its trigger
            self.on_end(cam, zid, ts, zs.peak)

        if st.bg is None:
            st.bg = gray.astype(np.float32)
            st.last_ts = ts
            return

        diff = cv2.absdiff(gray, cv2.convertScaleAbs(st.bg))
        cv2.accumulateWeighted(gray, st.bg, 0.05)
        if st.exclude is not None:
            diff[st.exclude > 0] = 0

        dt = max(ts - st.last_ts, 1e-3)
        st.last_ts = ts
        release_frames = max(1, int(self.release_seconds / dt))

        motion_zones = [z for z in zones if z.kind == "motion"] or [
            Zone(FULL_FRAME, cam, "Full frame", [[0, 0], [1, 0], [1, 1], [0, 1]])]
        for z in motion_zones:
            mask = st.masks.get(z.id)
            if mask is None:
                continue
            thr = 8 + (1.0 - z.sensitivity) * 42          # sensitivity 1.0 -> 8, 0.0 -> 50
            changed = (diff > thr) & (mask > 0)
            area = float(np.count_nonzero(changed)) / max(1, int(np.count_nonzero(mask)))
            s = st.zstate.setdefault(z.id, _ZoneState())
            if area >= z.min_area:
                s.hits += 1
                s.misses = 0
                s.peak = max(s.peak, area)
                if not s.active and s.hits >= z.min_frames:
                    s.active = True
                    self.on_start(cam, z.id, ts, area)
            else:
                s.hits = 0
                if s.active:
                    s.misses += 1
                    if s.misses >= release_frames:
                        s.active = False
                        self.on_end(cam, z.id, ts, s.peak)
                        s.peak = 0.0

    @staticmethod
    def _build_masks(st: _CamState, zones: list[Zone], shape: tuple[int, int]) -> None:
        h, w = shape
        st.masks.clear()
        st.shape = shape
        excl = np.zeros((h, w), np.uint8)
        has_excl = False
        motion = [z for z in zones if z.kind == "motion"]
        for z in zones:
            poly = (np.asarray(z.points, np.float32) * [w, h]).astype(np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            if z.kind == "exclude":
                excl |= m
                has_excl = True
            elif z.kind == "motion":
                st.masks[z.id] = m
        if not motion:
            st.masks[FULL_FRAME] = np.full((h, w), 255, np.uint8)
        st.exclude = excl if has_excl else None
