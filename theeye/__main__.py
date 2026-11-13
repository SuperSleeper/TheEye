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
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("libav").setLevel(logging.ERROR)

    cfg = load_config(args.config)
    hub = Hub(cfg)
    searcher = Searcher(cfg.llm, hub.db, {c.id: c.cfg.name or c.id for c in hub.cams.values()}, hub.embedder)
    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        logging.warning("%s not found — starting empty; open the Setup tab to configure", cfg_path)
    app = create_app(hub, searcher, cfg_path)
    # live-view streams never end on their own; without a timeout, shutdown waits for every open browser tab
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="warning", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
