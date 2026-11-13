"""Configuration loading. Cameras live in YAML; zones/rules live in the DB (edited via UI)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import yaml
from pydantic import BaseModel, Field, model_validator

_URL_CREDS = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://[^:/@\s]*):[^@\s]*@")


def redact(text: str) -> str:
    """Mask passwords in any URLs inside text (error messages from ffmpeg/PyAV include the full URL)."""
    return _URL_CREDS.sub(r"\1:********@", text)


# Brand presets: RTSP paths for main/sub streams. {ch} = channel number (1-based).
BRANDS: dict[str, dict] = {
    "hikvision": {"label": "Hikvision / Annke / Swann (Hik OEM)", "port": 554,
                  "main": "/Streaming/Channels/{ch}01", "sub": "/Streaming/Channels/{ch}02"},
    "dahua": {"label": "Dahua / Amcrest / Lorex (Dahua OEM)", "port": 554,
              "main": "/cam/realmonitor?channel={ch}&subtype=0", "sub": "/cam/realmonitor?channel={ch}&subtype=1"},
    "reolink": {"label": "Reolink", "port": 554,
                "main": "/h264Preview_{ch:02d}_main", "sub": "/h264Preview_{ch:02d}_sub"},
    "uniview": {"label": "Uniview (UNV)", "port": 554,
                "main": "/unicast/c{ch}/s0/live", "sub": "/unicast/c{ch}/s1/live"},
    "axis": {"label": "Axis", "port": 554,
             "main": "/axis-media/media.amp?camera={ch}", "sub": "/axis-media/media.amp?camera={ch}&resolution=640x360"},
    "tapo": {"label": "TP-Link Tapo / VIGI", "port": 554, "main": "/stream1", "sub": "/stream2"},
    "foscam": {"label": "Foscam", "port": 88, "main": "/videoMain", "sub": "/videoSub"},
    "custom": {"label": "Custom URL (any RTSP / RTMP / HTTP / UDP source)", "port": 554, "main": "", "sub": ""},
}


def build_url(brand: str, host: str, port: int, user: str, password: str, path: str) -> str:
    auth = ""
    if user:
        auth = quote(user, safe="") + (":" + quote(password, safe="") if password else "") + "@"
    return f"rtsp://{auth}{host}:{port}{path}"


class CameraConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$")
    name: str = ""
    # Either pick a brand + host/credentials (URLs are generated), or brand "custom" + explicit URLs.
    brand: str = "custom"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    channel: int = 1
    main_url: str = ""                 # high-res stream -> RAM ring buffer -> clips
    sub_url: Optional[str] = None      # low-res stream -> motion/detection (falls back to main_url)
    enabled: bool = True
    ring_seconds: int = 300            # length of the RAM loop
    analyze_fps: float = 5.0           # frames/s pushed through motion analysis
    detect_fps: float = 3.0            # frames/s pushed to the object detector while active
    always_detect: bool = False        # run detector even without motion

    @model_validator(mode="after")
    def _derive_urls(self) -> "CameraConfig":
        b = BRANDS.get(self.brand)
        if b is None:
            raise ValueError(f"unknown brand {self.brand!r}")
        if self.brand != "custom":
            if not self.host:
                raise ValueError("host is required for brand presets")
            port = self.port or b["port"]
            self.main_url = build_url(self.brand, self.host, port, self.username or "", self.password or "",
                                      b["main"].format(ch=self.channel))
            self.sub_url = build_url(self.brand, self.host, port, self.username or "", self.password or "",
                                     b["sub"].format(ch=self.channel))
        elif not self.main_url:
            raise ValueError("main_url is required for custom cameras")
        return self

    def to_yaml_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if self.brand != "custom":           # URLs are regenerated from host/credentials
            d.pop("main_url", None)
            d.pop("sub_url", None)
        return d


class DetectorConfig(BaseModel):
    enabled: bool = True
    model: str = "yolo11s.pt"
    device: str = "cuda:0"
    confidence: float = 0.45
    classes: list[str] = Field(default_factory=lambda: [
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "bird"])
    plates: bool = True
    plate_detector: str = "yolo-v9-t-384-license-plate-end2end"
    plate_ocr: str = "cct-xs-v2-global-model"
    embeddings: bool = True            # CLIP embeddings for visual search
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"


class LLMConfig(BaseModel):
    enabled: bool = True
    url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:4b"
    keep_alive: str = "5m"


class StorageConfig(BaseModel):
    path: str = "data"
    max_gb: float = 200.0              # oldest event clips pruned beyond this
    ram_budget_gb: float = 16.0        # hard cap across all ring buffers


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class Config(BaseModel):
    cameras: list[CameraConfig] = Field(default_factory=list)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @property
    def data_dir(self) -> Path:
        p = Path(self.storage.path)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():                      # first run: start empty, configure from the Setup page
        return Config()
    with open(p) as f:
        return Config.model_validate(yaml.safe_load(f) or {})


def save_config(cfg: Config, path: str | Path) -> None:
    """Write YAML atomically (keeps a .bak); file is 0600 because it holds camera passwords."""
    p = Path(path)
    d = cfg.model_dump(exclude_none=True)
    d["cameras"] = [c.to_yaml_dict() for c in cfg.cameras]
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write("# TheEye configuration (managed by the Setup page; hand edits are fine too)\n")
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True)
    os.chmod(tmp, 0o600)
    if p.exists():
        p.replace(p.with_suffix(p.suffix + ".bak"))
    tmp.replace(p)
