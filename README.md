# TheEye

A self-hosted NVR that keeps a rolling loop of every camera **in RAM** and only writes to disk when an
**event** fires. It includes object detection (people, vehicles, animals), licence-plate reading, zones
you draw on screen, and natural-language search powered by a local LLM.

```
 RTSP / RTMP / HTTP / UDP cameras
   ├─ main stream ─► demux only ─► RAM ring buffer (compressed packets, GOP-aligned)
   │                                     │  on event: pre-roll from RAM + live post-roll
   │                                     ▼
   │                                 Recorder ─► data/clips/*.mp4   (stream copy, no re-encode)
   └─ sub stream ──► decode ─► Motion engine (your zones) ─► motion_triggers
                            └► YOLO + tracker ─► plates (main-stream crop) ─► CLIP embedding ─► detections
                                        │
            motion triggers + detections ──► Rules engine ──► events ──► Recorder / webhook
                                                                      ▲
                       Search box ─► Ollama (text → JSON filter) ─► SQL + CLIP rerank
```

## Concepts

| Entity | Meaning |
|---|---|
| **Zone** | A polygon drawn on a camera. Kinds: **motion** (generates motion triggers and tags detections), **detect** (tags detections only), **exclude** (ignored by both motion and AI). |
| **Motion trigger** | A raw signal: "zone X had motion from t1 to t2". Always logged (Motion log tab) and never records on its own. |
| **Detection** | One tracked object: label, best snapshot, zones it entered, plate, embedding. |
| **Rule** | Camera scope + optional schedule + conditions (`motion` in zone, `object` of labels in zone, `plate` matching a glob), combined with ANY/ALL, plus pre-roll, post-roll, max length, record on/off, webhook. |
| **Event** | Created by a rule, a manual "Record" press, or `POST /api/hooks/{camera}`. Owns a clip offset, snapshot, labels and plates, and links to the triggers and detections involved. |

When several events overlap on one camera they share one clip file, and each event stores its own offset.

## Running

```bash
uv venv --system-site-packages -p python3.12 .venv && . .venv/bin/activate
uv pip install -r requirements.txt
ollama pull gemma3:4b                # if not already present
./run.sh                             # → https://localhost:8080
```

`run.sh` runs the code from this repo inside a separate **install folder** (default `~/TheEye-install`,
or `./run.sh /some/other/folder`, or `THEEYE_HOME=...`). Everything private lives there, outside git:

```
~/TheEye-install/
  config.yaml   cameras, logins, passwords (created from config.example.yaml on first run; edit on the Setup tab)
  data/         database, clips, snapshots
  tls/          self-signed HTTPS certificate and its private key (created on first start)
  auth.json     web login: username + salted scrypt hash of the password (never the password itself)
  yolo11s.pt    model weights (downloaded on first run)
```

It refuses an install folder inside the repo. Use a second folder (e.g. `./run.sh ~/TheEye-test`) for a
throwaway test install with its own config and data.

The first start downloads the YOLO, plate and CLIP weights (about 400 MB).

The web UI is served over **HTTPS** with a self-signed certificate made for this machine's names and addresses
(renewed automatically when they change or it nears expiry). Browsers warn the first time: choose *Advanced → Proceed*,
or import `tls/cert.pem` into your browser/OS as trusted to remove the warning. To use your own certificate, set
`server.cert_file` / `server.key_file`; to turn HTTPS off, untick it on the Setup tab (or `server.https: false`).

### Sign-in

The web UI asks for a username and password. Create the login the first time by opening the UI **on the TheEye
computer itself** (`https://localhost:8080`; other devices can't create it), or with `./run.sh --set-password`, which
also resets a forgotten password and signs everyone out. Change it later under Setup → Sign-in.

- Passwords are stored only as a salted scrypt hash in `auth.json` (install folder, mode 600).
- Sessions use a random token in an `HttpOnly`, `Secure`, `SameSite=Strict` cookie; the database keeps only its hash.
  "Keep me signed in" lasts 30 days, otherwise until the browser closes (at most 12 h).
- After 5 wrong passwords an address is locked out for 30 s, doubling each time up to 1 h.
- Changing the password signs out every other device; Setup also has "Sign out all other devices".
- Requests that change something are refused unless they come from TheEye's own pages.

### Keeping credentials out of git

Camera passwords live in the install folder, which is outside the repo. If a `config.yaml` or `data/` ends up in the
repo anyway, they're git-ignored, along with `.bak` backups.
As a second line of defence, hooks in `.githooks/` block any commit, commit message or push that adds:

- a forbidden file (`config.yaml` and backups, `data/`, `.env`, private keys)
- one of the real passwords from your `config.yaml` (in the repo or any install folder `run.sh` has used)
- personal details: your camera IPs/hostnames and non-generic camera names (from `config.yaml`),
  your git email, your home folder path, or any real-looking email address
- a non-placeholder `password:` in a config file, or a URL with an embedded `user:password@`

Use sample data instead: `192.168.1.50`, `user:pass`, `change-me`, `someone@example.com`, `Driveway`.
Turn the hooks on once per clone:

```bash
git config core.hooksPath .githooks
```

Check the whole history any time with `.githooks/check-secrets main`.

## Sizing (5–12 cameras on this box)

* **RAM:** about 45 MB per minute per 4 MP camera at 6 Mbit/s. With 12 cameras on a 10-minute loop that's about 5.4 GB.
  `ram_budget_gb` is a hard cap; if it's reached, the oldest GOPs are dropped first.
* **GPU (RTX 3070, 8 GB):** YOLO11s about 0.5 GB, CLIP about 0.6 GB, gemma3:4b about 4 GB (loaded only while searching).
  Inference is batched across cameras and took about 5 ms per batch in testing.
* **CPU:** sub-stream decoding (640×360) is cheap. Always configure `sub_url`; decoding 4 MP main streams for
  analysis would use far more CPU.
* Set each camera's **keyframe interval to 1–2 s** so pre-roll is precise and plate frames are fresh.

## API

| | |
|---|---|
| `GET /api/cameras`, `/api/cameras/{id}/live.mjpg`, `/snapshot.jpg`, `/state` | status, live view, current tracks and active zones |
| `GET/POST/PUT/DELETE /api/zones`, `/api/rules` | zone and rule management |
| `GET /api/events`, `/api/events/{id}` | events with linked triggers and detections |
| `POST /api/events/manual` `{camera_id, duration, preroll}` | record now |
| `POST /api/hooks/{camera_id}?duration=30` | external trigger (doorbell, alarm panel, Home Assistant); send `Authorization: Bearer <token>` from Setup → Sign-in |
| `GET /api/triggers`, `/api/detections` | raw motion log and objects |
| `POST /api/search` `{q}` | natural-language search |

## Known limitations / next steps

* Audio is not recorded (video-only clips). Many cameras send G.711, which MP4 handles poorly.
* Live view is MJPEG from the sub stream (simple, about 0.5 s latency). A WebRTC path (e.g. go2rtc) would give
  full-res, low-latency live view.
* The 4B LLM sometimes gets relative-time arithmetic wrong ("past 10 minutes"). Its output is checked
  against the question (cameras, zones and times must be mentioned), and the interpreted filter is shown
  above the results. Switching `llm.model` to a larger model improves this.
* ONNX Runtime GPU wheels expect CUDA 12 while this PyTorch ships CUDA 13, so plate models run on CPU (a few ms each).
* Face recognition (identifying *who*) is not included yet. The detection and embedding pipeline is the
  place to add it.
* No authentication: run it on a trusted LAN or behind a reverse proxy with auth.
