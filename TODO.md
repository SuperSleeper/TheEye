# TheEye – ToDo

## Home Assistant integration (like Frigate)

Goals:
1. **Don't add load on the cameras.** Many cameras (e.g. Wyze) cope badly with more than one stream, so Home
   Assistant must get video from TheEye, never from the camera directly.
2. **Motion and detections drive automations** (alarms, lights, devices).

Plan:
- [ ] **MQTT with Home Assistant auto-discovery** (about 1 day). A Setup card for broker host/port/login.
      Per camera: binary sensors for motion, person / vehicle / animal (optionally per zone) and recording;
      sound level; last plate; latest snapshot image. Each rule firing goes out as an event with camera, rule,
      labels, plate and a snapshot link. Sensors stay on while the thing is present, so they work for alarms.
- [ ] **Video relay** (about 1 day). TheEye pushes each camera's H.264 (copied, not re-encoded) to a local
      go2rtc, which serves `rtsp://<theeye-host>:8554/<camera>` with a login. Each camera keeps exactly one
      connection, to TheEye. TheEye starts and supervises go2rtc. Home Assistant's own go2rtc gives
      low-delay WebRTC in dashboards. Fallback if go2rtc is unwanted: an HTTP stream served by TheEye itself.
- [ ] Optional later: a custom HACS integration (clips in the HA media browser, search as an HA action).

Works today without code: a rule's webhook field can post to an HA automation's webhook trigger
(`http://<ha-host>:8123/api/webhook/<id>`). It fires once per event, with no "still present" state.
