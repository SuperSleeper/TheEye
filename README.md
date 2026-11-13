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
cp config.example.yaml config.yaml   # add your cameras
ollama pull gemma3:4b                # if not already present
python -m theeye -c config.yaml      # → http://localhost:8080
```

The first start downloads the YOLO, plate and CLIP weights (about 400 MB).

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
| `POST /api/hooks/{camera_id}?duration=30` | external trigger (doorbell, alarm panel, Home Assistant) |
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
