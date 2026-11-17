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
# Optional keys: "scheme" (default rtsp), "note" (shown in the Setup form). An empty "sub" means the
# camera has no low-res stream, so analysis decodes the main stream.
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
    # Wyze's built-in RTSP (Cam v3 4.36.16.x / Pan v3 4.55.16.x and later; also seen on Cam v4). RTSPS is the app default;
    # plain RTSP has to be enabled by accepting a security warning in the app. Single stream only.
    "wyze": {"label": "Wyze Cam v3 / v4 / Pan v3 — RTSPS (secure)", "port": 322, "scheme": "rtsps",
             "main": "/stream0", "sub": "",
             "note": "Create the login in the Wyze app: camera → Settings → Advanced Settings → RTSP → Set Up "
                     "(it is separate from your Wyze account). Wyze has a single stream, so motion/AI decode the "
                     "full-resolution stream (more CPU). Models without RTSP in the app (e.g. Cam OG) can "
                     "go through docker-wyze-bridge with Custom URL."},
    "wyze_rtsp": {"label": "Wyze Cam v3 / v4 / Pan v3 — plain RTSP", "port": 554, "main": "/stream0", "sub": "",
                  "note": "Requires turning off Secure RTSP (RTSPS) in the Wyze app's RTSP settings. Prefer the "
                          "RTSPS option unless a client can't do TLS. Single stream only."},
    "custom": {"label": "Custom URL (any RTSP / RTMP / HTTP / UDP source)", "port": 554, "main": "", "sub": ""},
}


def build_url(brand: str, host: str, port: int, user: str, password: str, path: str) -> str:
    auth = ""
    if user:
        auth = quote(user, safe="") + (":" + quote(password, safe="") if password else "") + "@"
    scheme = BRANDS.get(brand, {}).get("scheme", "rtsp")
    return f"{scheme}://{auth}{host}:{port}{path}"


class Credential(BaseModel):
    """A saved login that several cameras can share; changing it updates every camera using it."""
    name: str = Field(min_length=1, max_length=60)
    username: str = ""
    password: Optional[str] = None


def resolve_credentials(cams: list, creds: list) -> list:
    """Fill username/password of cameras that reference a saved credential (raw dicts, before validation)."""
    by_name = {}
    for c in creds or []:
        c = c.model_dump() if isinstance(c, BaseModel) else c
        by_name[c.get("name")] = c
    out = []
    for cam in cams or []:
        cam = cam.model_dump() if isinstance(cam, BaseModel) else dict(cam)
        ref = cam.get("credential")
        if ref:
            if ref not in by_name:
                raise ValueError(f"camera {cam.get('id')!r} uses unknown saved login {ref!r}")
            cam["username"] = by_name[ref].get("username") or None
            cam["password"] = by_name[ref].get("password")
        out.append(cam)
    return out


class CameraConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,39}$")
    name: str = ""
    # Either pick a brand + host/credentials (URLs are generated), or brand "custom" + explicit URLs.
    brand: str = "custom"
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    credential: Optional[str] = None   # name of a saved Credential; overrides username/password
    channel: int = 1
    main_url: str = ""                 # high-res stream -> RAM ring buffer -> clips
    sub_url: Optional[str] = None      # low-res stream -> motion/detection (falls back to main_url)
    enabled: bool = True
    ring_seconds: int = 300            # length of the RAM loop
    analyze_fps: float = 5.0           # frames/s pushed through motion analysis
    detect_fps: float = 3.0            # frames/s pushed to the object detector while active
    always_detect: bool = False        # run detector even without motion
    record_audio: bool = True          # include the camera's audio track in clips
    sound_detect: bool = False         # a sound louder than sound_threshold_db starts an event (recording)
    sound_threshold_db: float = Field(-30.0, ge=-90, le=0)     # dBFS; 0 = loudest the mic can record
    sound_min_seconds: float = Field(0.5, ge=0.1, le=60)       # must stay that loud this long

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
                                     b["sub"].format(ch=self.channel)) if b["sub"] else None
        elif not self.main_url:
            raise ValueError("main_url is required for custom cameras")
        return self

    def to_yaml_dict(self) -> dict:
        d = self.model_dump(exclude_none=True)
        if self.brand != "custom":           # URLs are regenerated from host/credentials
            d.pop("main_url", None)
            d.pop("sub_url", None)
        if self.credential:                  # login lives in the shared credential
            d.pop("username", None)
            d.pop("password", None)
        return d


class DetectorConfig(BaseModel):
    enabled: bool = True
    model: str = "yolo11s.pt"
    device: str = "cuda:0"
    confidence: float = 0.45
    stationary_seconds: float = 30.0   # objects that haven't moved this long are "parked": rules ignore them (0 = off)
    require_motion: bool = True        # objects only trigger rules once motion was seen inside their box
    motion_min_fraction: float = 0.10  # share of the box that must change to count as "seen moving"
                                       # (a walking person changes far more; traffic behind a box, leaves less)
    classes: list[str] = Field(default_factory=lambda: [
        "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "bird"])
    plates: bool = True
    plate_detector: str = "yolo-v9-t-384-license-plate-end2end"
    plate_ocr: str = "cct-xs-v2-global-model"
    plates_refine: bool = False        # (off until tuned) read plates again over the saved recording when the machine is quiet
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
    max_gb: float = 200.0              # "max useable": oldest event clips pruned beyond this
    ram_budget_gb: float = 16.0        # hard cap across all ring buffers
    # Each camera's share of max_gb, in percent (camera id -> %). Cameras not listed split what's left equally.
    # A camera's oldest clips are deleted once it's past its share.
    camera_share: dict[str, float] = Field(default_factory=dict)

    def clip_quotas(self, cam_ids: list[str]) -> dict[str, float]:
        """Bytes of clips each camera may keep. Set shares adding up to more than 100% are scaled down to fit."""
        limit = self.max_gb * 1e9
        set_ = {c: max(0.0, float(self.camera_share[c])) for c in cam_ids
                if self.camera_share.get(c) is not None}
        total = sum(set_.values())
        if total > 100:
            set_ = {c: v * 100 / total for c, v in set_.items()}
            total = 100.0
        rest = [c for c in cam_ids if c not in set_]
        each = (100 - total) / len(rest) if rest else 0.0
        return {c: limit * (set_[c] if c in set_ else each) / 100 for c in cam_ids}


class VehiclesConfig(BaseModel):
    enabled: bool = True
    repeat_visits: int = Field(3, ge=2, le=100)        # flag a vehicle seen on this many separate visits
    visit_gap_minutes: float = Field(30, ge=1)         # sightings closer together than this are one visit
    match_similarity: float = Field(0.88, ge=0.5, le=0.99)   # appearance match threshold (cosine)
    labels: list[str] = Field(default_factory=lambda: ["car", "truck", "motorcycle", "bus"])


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    https: bool = True                 # serve over HTTPS; a self-signed certificate is created if none is given
    cert_file: Optional[str] = None    # your own certificate + key (PEM); relative paths are from the config folder
    key_file: Optional[str] = None
    mcp_port: Optional[int] = None     # if set, a read-only MCP server for local AI assistants on 127.0.0.1:<port>


class AnimalsConfig(BaseModel):
    """Name the animal more precisely than the detector can (it only knows bird/cat/dog), using the CLIP
    picture fingerprint every track already gets. Applies live; needs detector.embeddings on."""
    enabled: bool = True
    min_confidence: float = Field(0.5, ge=0.2, le=0.95)   # below this the detector's own label is kept
    # Step 1: which animal it really is. A squirrel the detector called "bird" is caught here.
    kinds: list[str] = Field(default_factory=lambda: [
        "bird", "squirrel", "cat", "dog", "raccoon", "opossum", "rabbit", "fox", "coyote", "deer", "skunk"])
    # Your own pets, by breed name (e.g. [pomeranian]). CLIP's generic descriptions can be confidently wrong about
    # one particular animal (an orange Pomeranian reads as "cat"), so a dog/cat that looks like this breed's own
    # past confident sightings (detector and CLIP both said dog, and this breed) is named that, whatever else.
    pets: list[str] = Field(default_factory=list)
    pet_match: float = Field(0.75, ge=0.5, le=0.99)       # similarity to the pet's 5th-closest past sighting
    # Step 2: which kind of bird / dog / cat.
    birds: list[str] = Field(default_factory=lambda: [
        "crow", "raven", "hawk", "owl", "pigeon", "mourning dove", "seagull", "sparrow", "finch", "robin",
        "blue jay", "cardinal", "mockingbird", "hummingbird", "woodpecker", "duck", "goose", "chicken", "parrot"])
    dogs: list[str] = Field(default_factory=lambda: [
        "pomeranian", "chihuahua", "labrador retriever", "golden retriever", "german shepherd", "pit bull",
        "husky", "poodle", "beagle", "dachshund", "corgi", "shih tzu", "yorkshire terrier", "border collie",
        "boxer", "bulldog", "french bulldog", "great dane", "rottweiler", "australian shepherd", "maltese", "pug",
        "samoyed", "american eskimo dog", "japanese spitz"])
    cats: list[str] = Field(default_factory=lambda: [
        "tabby cat", "siamese cat", "persian cat", "maine coon cat", "bengal cat"])   # colour-only names
                                                                    # ("white cat") beat breeds on white dogs


class DeliveryConfig(BaseModel):
    """Tag Amazon / UPS / FedEx / USPS vehicles (and, from a large clear view, uniformed people), using the CLIP
    fingerprint every track already gets. Applies live; needs detector.embeddings on."""
    enabled: bool = True
    min_confidence: float = Field(0.95, ge=0.5, le=0.999)   # tested on 2 days of footage: 13/13 right at 0.95


class PeopleConfig(BaseModel):
    """Recognise named people (family members) by face. Face models live in <install>/models."""
    enabled: bool = True


class Config(BaseModel):
    credentials: list[Credential] = Field(default_factory=list)
    cameras: list[CameraConfig] = Field(default_factory=list)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    vehicles: VehiclesConfig = Field(default_factory=VehiclesConfig)
    animals: AnimalsConfig = Field(default_factory=AnimalsConfig)
    delivery: DeliveryConfig = Field(default_factory=DeliveryConfig)
    people: PeopleConfig = Field(default_factory=PeopleConfig)

    @model_validator(mode="before")
    @classmethod
    def _apply_credentials(cls, data):
        if isinstance(data, dict) and data.get("cameras"):
            names = [c.get("name") if isinstance(c, dict) else c.name for c in data.get("credentials") or []]
            if len(names) != len(set(names)):
                raise ValueError("saved login names must be unique")
            data = {**data, "cameras": resolve_credentials(data["cameras"], data.get("credentials"))}
        return data

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
