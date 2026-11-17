"""Entry point:  python -m theeye -c config.yaml"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .api import create_app
from .config import load_config
from .hub import Hub
from .search import Searcher


def main() -> None:
    ap = argparse.ArgumentParser(prog="theeye")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--set-password", action="store_true", help="create the web login, or reset its password")
    args = ap.parse_args()
    if args.set_password:
        from .auth import set_password_cli
        cfg_path = Path(args.config).resolve()
        db = Path(load_config(cfg_path).storage.path)
        set_password_cli(cfg_path.parent / "auth.json", (cfg_path.parent / db if not db.is_absolute() else db) / "theeye.db")
        return
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("libav").setLevel(logging.ERROR)

    cfg = load_config(args.config)
    hub = Hub(cfg)
    searcher = Searcher(cfg.llm, hub.db, {c.id: c.cfg.name or c.id for c in hub.cams.values()}, hub.embedder,
                        hub.animals)
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        logging.warning("%s not found — starting empty; open the Setup tab to configure", cfg_path)
    hub.log_path = cfg_path.parent / "theeye.log"    # where run.sh / the service append the log
    from .people import People
    hub.people = People(hub, cfg_path.parent / "models")
    searcher.people = hub.people
    from .pets import Pets
    hub.pets = searcher.pets = Pets(hub)
    from .exhaust import ExhaustWatch
    hub.exhaust = ExhaustWatch(hub, cfg_path.parent / "models")
    app = create_app(hub, searcher, cfg_path)
    srv = cfg.server
    tls = {}
    if srv.https:
        if srv.cert_file and srv.key_file:
            cert, key = (cfg_path.parent / srv.cert_file, cfg_path.parent / srv.key_file)
        else:                       # kept next to the config (the install folder), never in the repo
            from .tls import ensure_cert
            cert, key = ensure_cert(cfg_path.parent / "tls")
        from .tlsredirect import make_protocol       # TLS on the port, and plain http there redirects to https
        # asyncio's own loop: uvloop starts reading a connection even while it's paused, which defeats the peek
        tls = {"http": make_protocol(str(cert), str(key)), "loop": "asyncio"}
    logging.info("web UI on %s://%s:%d", "https" if srv.https else "http",
                 "localhost" if srv.host in ("0.0.0.0", "::") else srv.host, srv.port)
    if srv.mcp_port:                 # read-only MCP for a local AI assistant (loopback, hook-token guarded)
        from . import mcp
        from .auth import AuthStore
        _store = AuthStore(cfg_path.parent / "auth.json", hub.db)
        mcp.serve("127.0.0.1", int(srv.mcp_port), hub, searcher, hub.db, _store.hook_token_ok)
    # live-view streams never end on their own; without a timeout, shutdown waits for every open browser tab
    uvicorn.run(app, host=srv.host, port=srv.port, log_level="warning", timeout_graceful_shutdown=3, **tls)


if __name__ == "__main__":
    main()
