"""SQLite persistence. Thread-safe via a single connection + lock (WAL mode)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Polygons drawn in the UI. points = [[x,y],...] normalized 0..1
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    name TEXT NOT NULL,
    points TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'motion',        -- motion | detect | exclude
    sensitivity REAL NOT NULL DEFAULT 0.5,      -- 0..1, higher = more sensitive
    min_area REAL NOT NULL DEFAULT 0.02,        -- fraction of zone that must change
    min_frames INTEGER NOT NULL DEFAULT 2,      -- consecutive analysed frames
    color TEXT NOT NULL DEFAULT '#ff3b30',
    enabled INTEGER NOT NULL DEFAULT 1
);

-- Motion triggers: raw signals. Logged independently of events.
CREATE TABLE IF NOT EXISTS motion_triggers (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    zone_id INTEGER,
    start_ts REAL NOT NULL,
    end_ts REAL,
    peak_area REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS mt_cam_ts ON motion_triggers(camera_id, start_ts);

-- Rules map trigger conditions -> event creation.
CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    camera_id TEXT NOT NULL DEFAULT '*',
    enabled INTEGER NOT NULL DEFAULT 1,
    match TEXT NOT NULL DEFAULT 'any',          -- any | all
    conditions TEXT NOT NULL,                   -- JSON list, see rules.py
    preroll REAL NOT NULL DEFAULT 10,
    postroll REAL NOT NULL DEFAULT 10,
    max_length REAL NOT NULL DEFAULT 600,
    schedule TEXT,                              -- JSON {days:[0-6], start:"HH:MM", end:"HH:MM"}
    record INTEGER NOT NULL DEFAULT 1,
    webhook TEXT
);

-- Events: recorded incidents. Separate entity from triggers.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    rule_id INTEGER,
    source TEXT NOT NULL DEFAULT 'rule',        -- rule | manual | api
    start_ts REAL NOT NULL,
    end_ts REAL,
    clip_path TEXT,
    clip_offset REAL NOT NULL DEFAULT 0,       -- seconds into clip where this event's pre-roll starts
    snapshot_path TEXT,
    labels TEXT NOT NULL DEFAULT '[]',
    plates TEXT NOT NULL DEFAULT '[]',
    note TEXT
);
CREATE INDEX IF NOT EXISTS ev_cam_ts ON events(camera_id, start_ts);

-- Loudness history for tuning sound triggers: one row per camera per 10 s.
CREATE TABLE IF NOT EXISTS sound_levels (
    camera_id TEXT NOT NULL,
    ts REAL NOT NULL,                           -- start of the 10 s bucket
    peak_db REAL NOT NULL,                      -- loudest 100 ms window, dBFS
    avg_db REAL NOT NULL                        -- mean of the windows, dBFS
);
CREATE INDEX IF NOT EXISTS snd_cam_ts ON sound_levels(camera_id, ts);

-- Repeat-vehicle tracking: one row per distinct vehicle (matched by plate or appearance).
CREATE TABLE IF NOT EXISTS vehicles (
    id INTEGER PRIMARY KEY,
    grp TEXT NOT NULL,                          -- matching group: car (car/truck/bus) | motorcycle | ...
    label TEXT NOT NULL,                        -- label of the first sighting
    plate TEXT,
    embedding BLOB,                             -- n_ex exemplar sighting embeddings (float32, row-major)
    n_ex INTEGER NOT NULL DEFAULT 0,
    sightings INTEGER NOT NULL DEFAULT 1,
    visits INTEGER NOT NULL DEFAULT 1,          -- sightings separated by more than the visit gap
    first_ts REAL NOT NULL,
    last_ts REAL NOT NULL,
    cameras TEXT NOT NULL DEFAULT '[]',
    snapshot_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',         -- new | flagged | known
    flagged_ts REAL,
    note TEXT
);
CREATE INDEX IF NOT EXISTS veh_status ON vehicles(status, visits);

-- Which triggers / detections caused or occurred during an event.
CREATE TABLE IF NOT EXISTS event_links (
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                         -- motion | detection
    ref_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS el_ev ON event_links(event_id);

-- Object detections (one row per tracked object, updated with best snapshot).
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY,
    camera_id TEXT NOT NULL,
    track_key TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    first_ts REAL NOT NULL,
    last_ts REAL NOT NULL,
    bbox TEXT NOT NULL,                         -- normalized [x1,y1,x2,y2] of best frame
    zones TEXT NOT NULL DEFAULT '[]',
    plate TEXT,
    plate_conf REAL,
    snapshot_path TEXT,
    embedding BLOB,
    event_id INTEGER
);
CREATE INDEX IF NOT EXISTS det_cam_ts ON detections(camera_id, first_ts);
CREATE INDEX IF NOT EXISTS det_label ON detections(label);
CREATE INDEX IF NOT EXISTS det_plate ON detections(plate);
"""


class DB:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self._add_columns()

    # Columns added after the first release. CREATE TABLE IF NOT EXISTS won't add them to existing DBs.
    ADDED_COLUMNS = {
        "events": {
            "trigger_ts": "REAL",                         # when the rule fired (start_ts = trigger_ts - preroll)
            "thumb_score": "REAL NOT NULL DEFAULT 0",     # quality of the object-centered thumbnail in use
        },
        "zones": {
            "ai": "INTEGER NOT NULL DEFAULT 1",           # objects only count when their base is inside such a zone
        },
        "detections": {
            "vehicle_id": "INTEGER",                      # repeat-vehicle tracking
            "moving": "INTEGER",                          # was seen moving (NULL = recorded before this was stored)
        },
    }

    def _add_columns(self) -> None:
        for table, cols in self.ADDED_COLUMNS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in cols.items():
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.conn.execute(sql, tuple(params))

    def insert(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.lock:
            return self.conn.execute(sql, tuple(params)).lastrowid

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        with self.lock:
            r = self.conn.execute(sql, tuple(params)).fetchone()
            return dict(r) if r else None


def jload(s: str | None, default=None):
    if s is None:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default
