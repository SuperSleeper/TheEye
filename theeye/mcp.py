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


class _Tools:
    """The read-only tool surface. Each method returns a plain-text string for the assistant to read."""

    def __init__(self, hub: Any, searcher: Any, db: Any) -> None:
        self.hub = hub
        self.searcher = searcher
        self.db = db

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
