// TheEye NVR UI — vanilla JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts,
    body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body });
  if (r.status === 401) { location.replace('/login'); throw new Error('Signed out'); }
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtTime = ts => new Date(ts * 1000).toLocaleString();
const fmtDur = s => s == null ? '…' : s < 60 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
const LABELS = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'dog', 'cat', 'bird'];

let cameras = [];
let zonesCache = [];
const camName = id => (cameras.find(c => c.id === id) || {}).name || id;

window.addEventListener('error', e => {
  const b = document.querySelector('#cfg-banner');
  if (b) b.innerHTML = `<div class="banner err"><div>Page error: ${esc(e.message)} — try a hard refresh (Ctrl+Shift+R).</div></div>`;
});
window.addEventListener('unhandledrejection', e => console.error('Unhandled:', e.reason));

// ---------------------------------------------------------------- tabs
const timers = {};
function showTab(name) {
  $$('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  $$('.tab').forEach(s => s.classList.toggle('active', s.id === `tab-${name}`));
  Object.values(timers).forEach(clearInterval);
  for (const k in timers) delete timers[k];
  if (name !== 'live') { $('#live-grid').innerHTML = ''; closeFocus?.(); }
  queueMicrotask(() => lvUpdate());
  ({ live: initLive, events: loadEvents, triggers: loadTriggers, zones: initZones, rules: initRules, setup: () => initSetup(), search: () => { renderSearchCams(); $('#search-q').focus(); }, vehicles: loadVehicles })[name]?.();
  location.hash = name;
  $('#overlay-toggles').hidden = name !== 'live';       // overlays only exist on Live
}
$$('#tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));

async function boot() {
  // Endless MJPEG responses would otherwise hold the page's load event open forever.
  if (document.readyState !== 'complete') await new Promise(r => addEventListener('load', r, { once: true }));
  cameras = await api('/api/cameras');
  zonesCache = await api('/api/zones');
  refreshCameraSelects();
  setInterval(loadStats, 3000); loadStats();
  showTab(cameras.length ? (location.hash.slice(1) || 'live') : 'setup');   // first run → configurator
}

async function loadStats() {
  try {
    const s = await api('/api/stats');
    const rec = Object.keys(s.recording || {}).length;
    $('#stats').textContent = `RAM loop ${(s.ram_bytes / 1e9).toFixed(2)} / ${(s.ram_budget / 1e9).toFixed(0)} GB · ` +
      `${rec} recording · ${s.open_events} open events · ` +
      (s.detector ? `AI ${s.detector.infer_ms} ms · ` : '') + `disk free ${(s.disk_free / 1e9).toFixed(0)} GB`;
  } catch { $('#stats').textContent = 'offline'; }
}

// ---------------------------------------------------------------- live
// Which Live overlays to draw. Per-browser preference; storage may be unavailable (private mode etc.).
const OV = (() => {
  try { return { ai: true, zones: true, ...JSON.parse(localStorage.getItem('theeye.overlays') || '{}') }; }
  catch { return { ai: true, zones: true }; }
})();
function syncOverlayButtons() {
  $$('button[data-ov]').forEach(b => b.setAttribute('aria-pressed', String(!!OV[b.dataset.ov])));
}
$$('button[data-ov]').forEach(b => b.onclick = () => {
  OV[b.dataset.ov] = !OV[b.dataset.ov];
  try { localStorage.setItem('theeye.overlays', JSON.stringify(OV)); } catch { /* not persisted */ }
  syncOverlayButtons();
});
syncOverlayButtons();

function drawOverlay(cv, img, st, camId) {
  const w = cv.width = img.clientWidth, h = cv.height = img.clientHeight;
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  // object-fit: contain → compute drawn image rect
  const ar = (img.naturalWidth || img.videoWidth || 16) / (img.naturalHeight || img.videoHeight || 9);
  let dw = w, dh = w / ar; if (dh > h) { dh = h; dw = h * ar; }
  const ox = (w - dw) / 2, oy = (h - dh) / 2;
  for (const z of OV.zones ? zonesCache.filter(z => z.camera_id === camId && z.enabled) : []) {
    const active = st.motion_zones.includes(z.id);
    ctx.beginPath();
    z.points.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](ox + x * dw, oy + y * dh));
    ctx.closePath();
    ctx.strokeStyle = z.color; ctx.lineWidth = active ? 3 : 1; ctx.setLineDash(z.kind === 'exclude' ? [4, 4] : []);
    ctx.stroke();
    if (active) { ctx.fillStyle = z.color + '44'; ctx.fill(); }
  }
  ctx.setLineDash([]);
  ctx.font = '12px system-ui';
  for (const t of OV.ai ? st.tracks : []) {
    const [x1, y1, x2, y2] = t.bbox;
    // Ignored by rules: parked (grey dashed) or never seen moving (grey dotted)
    const ignored = t.parked || t.confirmed === false, col = ignored ? '#9aa3ad' : '#ffcc00';
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(t.parked ? [6, 4] : t.confirmed === false ? [2, 3] : []);
    ctx.strokeRect(ox + x1 * dw, oy + y1 * dh, (x2 - x1) * dw, (y2 - y1) * dh);
    ctx.setLineDash([]);
    const tag = t.parked ? ' · parked' : t.confirmed === false ? ` ${(t.conf * 100).toFixed(0)}% · no motion` : ` ${(t.conf * 100).toFixed(0)}%`;
    const label = `${t.label}${tag}${t.plate ? ' ' + t.plate : ''}`;
    ctx.fillStyle = col; ctx.fillRect(ox + x1 * dw, oy + y1 * dh - 16, ctx.measureText(label).width + 8, 16);
    ctx.fillStyle = '#000'; ctx.fillText(label, ox + x1 * dw + 4, oy + y1 * dh - 4);
  }
}

function initLive() {
  const grid = $('#live-grid');
  grid.innerHTML = cameras.length ? cameras.map(c => `
    <div class="tile" data-cam="${esc(c.id)}">
      <div style="position:relative"><img alt="" draggable="false">${LIVE_VIDEO_ENABLED ? '<video muted playsinline disablepictureinpicture></video>' : ''}<canvas></canvas><div class="tile-msg">Connecting…</div></div>
      <div class="bar">
        <div class="bar-top"><span class="grip" title="Drag to reorder">⠿</span><span class="name" title="${esc(c.name)}">${esc(c.name)}</span>
          <span class="badge st"></span><span class="badge recb hidden rec">REC</span>
          <button class="recnow" title="Record the next 30 seconds">● Record 30s</button>
          <button class="cfg" title="Edit this camera in Setup">⚙</button></div>
        <div class="bar-sub"><span class="ring"></span><span class="lat hidden"></span>
          <span class="snd hidden" title="Sound level right now (dBFS)"><i><b></b></i><span></span></span></div></div>
    </div>`).join('') : '<p class="muted">No cameras configured — add them on the Setup tab.</p>';
  lvReset(grid);
  $$('.tile', grid).forEach(tile => {
    $('.cfg', tile).onclick = () => editCamera(tile.dataset.cam);
    tile.firstElementChild.ondblclick = () => openFocus(tile.dataset.cam);
    tile.firstElementChild.title = 'Double-click for full-resolution view';
    $('.grip', tile).onpointerdown = e => startTileDrag(e, tile, grid);
    pumpFrames(tile);
    $('.recnow', tile).onclick = async () => {
      await api('/api/events/manual', { method: 'POST', body: { camera_id: tile.dataset.cam, duration: 30 } });
    };
  });
  const tick = async () => {
    const cams = await api('/api/cameras').catch(() => []);
    for (const tile of $$('.tile', grid)) {
      const id = tile.dataset.cam;
      const c = cams.find(x => x.id === id); if (!c) continue;
      const st = $('.st', tile);
      const live = c.status.main === 'live';
      st.textContent = live ? c.status.main_codec : c.status.main;
      st.className = 'badge st ' + (live ? 'ok' : 'err');
      const msg = $('.tile-msg', tile);
      msg.classList.toggle('hidden', live && tile.dataset.frames === '1');
      if (!live) msg.textContent = friendlyCamError(c.status.main_err, c.status.main);
      $('.ring', tile).textContent = `loop ${Math.floor(c.ring.seconds)}s · ${(c.ring.bytes / 1e6).toFixed(0)} MB`;
      showLatency(tile, c, live);
      const snd = $('.snd', tile);
      snd.classList.toggle('hidden', c.sound_db == null);
      if (c.sound_db != null) {
        $('b', snd).style.width = `${Math.max(0, Math.min(100, (c.sound_db + 90) / 90 * 100))}%`;   // -90..0 dBFS
        $(':scope > span', snd).textContent = `${Math.round(c.sound_db)} dB`;
      }
      api(`/api/cameras/${encodeURIComponent(id)}/state`).then(s => {
        $('.recb', tile).classList.toggle('hidden', !s.recording);
        const hist = tile._st ||= [];
        hist.push({ t: performance.now(), s }); while (hist.length > 30) hist.shift();
        const vid = tile.dataset.vid === '1';
        let shown = s;
        if (vid) {          // boxes come from live detection; the video runs behind, so show the boxes from back then
          const at = performance.now() - lvOverlayDelay(tile, c) * 1000;
          shown = [...hist].reverse().find(h => h.t <= at)?.s ?? hist[0].s;
        }
        drawOverlay($('canvas', tile), vid ? $('video', tile) : $('img', tile), shown, id);
      }).catch(() => {});
    }
  };
  timers.live = setInterval(() => { tick(); lvTick(); }, 500); tick();
}

// How far behind real time the tile's picture is. Per frame: the server's X-Frame-Age (stream lag when the
// frame arrived + time since) plus this browser's fetch time. Shows the worst of the last 2 s so it doesn't flicker.
const LAT_WINDOW_MS = 2000;
function showLatency(tile, c, live) {
  const el = $('.lat', tile), now = performance.now();
  const samples = (tile._lat || []).filter(x => now - x.t < LAT_WINDOW_MS);
  tile._lat = samples;
  const stream = c.lag?.[c.feed];
  let total = samples.length ? Math.max(...samples.map(x => x.age + x.net)) : stream;   // no recent frames: server's view
  el.classList.toggle('hidden', !live || total == null);
  if (!live || total == null) return;
  const net = samples.length ? Math.max(...samples.map(x => x.net)) : null;
  const buf = samples.some(x => x.buf != null) ? Math.max(...samples.map(x => x.buf || 0)) : null;
  const fmt = v => v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(1)} s`;
  el.textContent = `⏱ ${fmt(total)}`;
  el.className = 'lat' + (total >= 3 ? ' bad' : total >= 1 ? ' warn' : '');
  el.title = `Live picture is about ${fmt(total)} behind real time.\n`
    + (stream != null ? `• Camera → server: +${fmt(stream)} over its best in the last minute\n` : '')
    + (net != null ? `• Server → this browser: ${fmt(net)}\n` : '')
    + (buf != null ? `• Video buffered in this browser: ${fmt(buf)}\n` : '')
    + (c.feed === 'sub' && c.lag?.main != null ? `• Recording (main) stream: +${fmt(c.lag.main)}\n` : '')
    + `The camera's own encoding/network delay isn't included: these cameras don't send a clock to measure it against.`;
}

// ---- live video tiles
// Snapshots by default on every device. Video crashed some browsers when every tile played at once (Chrome on
// Linux/NVIDIA, iPad Safari), so it's opt-in via the ▶ Video button, remembered per browser. If the page dies
// while video is on, the next load comes back on snapshots (see LV_ARMED).
// All tiles share ONE WebSocket (see /api/live/ws), because browsers allow only ~6 connections per site.
// A tile shows its snapshot until its video actually plays, so it never goes blank.
const LIVE_VIDEO_ENABLED = true;
// Chrome on Linux with an NVIDIA GPU: decoding many tiles at once crashed Chrome's media process and then the
// nvidia-drm display driver ("Failed to apply atomic modeset"), hanging the whole machine, not just the tab.
// Not worth a retry, so video tiles are blocked there outright (the single full-window video still works).
const GPU_BLOCKS_VIDEO = (() => {
  const ua = navigator.userAgent;
  if (!/Linux/.test(ua) || /Android/.test(ua)) return false;
  try {
    const gl = document.createElement('canvas').getContext('webgl');
    if (!gl) return false;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const r = String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /nvidia/i.test(r);
  } catch { return false; }
})();
const CAN_VIDEO = LIVE_VIDEO_ENABLED && !GPU_BLOCKS_VIDEO && !!(window.ManagedMediaSource || window.MediaSource);
const LV_ARMED = 'theeye.liveVideoRunning';
const LV = { mode: 'snap', crashed: false, ws: null, tiles: new Map(), visible: new Set(), sent: '', rtt: 0, io: null, ping: null, backoff: 1 };
LV.mode = (() => {
  if (!CAN_VIDEO) return 'snap';
  let m = null, crashed = false;
  try {
    m = localStorage.getItem('theeye.liveMode');
    crashed = !!localStorage.getItem(LV_ARMED);    // set while video plays, cleared when the page closes normally
    if (crashed) { localStorage.setItem('theeye.liveMode', 'snap'); localStorage.removeItem(LV_ARMED); }
  } catch { /* private mode */ }
  if (crashed) { LV.crashed = true; return 'snap'; }
  return m === 'video' ? 'video' : 'snap';
})();
// Crash guard: mark "video running" while any tile streams, unmark on a normal page close.
function lvArm(on) { try { on ? localStorage.setItem(LV_ARMED, String(Date.now())) : localStorage.removeItem(LV_ARMED); } catch { /* ignore */ } }
addEventListener('pagehide', () => lvArm(false));
function syncLiveModeButton() {
  const b = $('#live-mode');
  b.setAttribute('aria-pressed', String(LV.mode === 'video'));
  b.hidden = !CAN_VIDEO;                // not offered where it can't (or mustn't) play
  b.title = !LIVE_VIDEO_ENABLED ? 'Video tiles are turned off'
    : GPU_BLOCKS_VIDEO ? 'Video tiles are off on Linux with an NVIDIA GPU: playing them hung the whole machine (display driver crash). Double-click a tile for full-window video'
    : LV.crashed && LV.mode === 'snap' ? 'Switched back to snapshots: the page closed unexpectedly while video tiles were playing. Click to try video again'
    : !CAN_VIDEO ? "This browser can't play live video tiles (no Media Source support)"
    : LV.mode === 'video' ? 'Live tiles show video. Click for snapshots (less data, better for phones)'
    : 'Live tiles show snapshots. Click for full-frame-rate video (more data)';
}
$('#live-mode').onclick = () => {
  LV.mode = LV.mode === 'video' ? 'snap' : 'video';
  try { localStorage.setItem('theeye.liveMode', LV.mode); } catch { /* not persisted */ }
  syncLiveModeButton(); lvUpdate();
};
syncLiveModeButton();
document.addEventListener('visibilitychange', () => lvUpdate());

function lvReset(grid) {                 // grid rebuilt: forget old tiles, watch which new ones are on screen
  for (const id of [...LV.tiles.keys()]) lvTeardown(id);
  LV.visible.clear(); LV.io?.disconnect();
  if (!CAN_VIDEO) return;
  LV.io = new IntersectionObserver(entries => {
    for (const e of entries) LV.visible[e.isIntersecting ? 'add' : 'delete'](e.target.dataset.cam);
    lvUpdate();
  });
  $$('.tile', grid).forEach(t => LV.io.observe(t));
}
function lvWanted() {
  if (LV.mode !== 'video' || document.hidden || F.cam || !$('#tab-live').classList.contains('active')) return [];
  return [...LV.visible].filter(id => $(`#live-grid .tile[data-cam="${CSS.escape(id)}"]`)).sort();
}
function lvUpdate() {
  const ids = lvWanted();
  for (const id of [...LV.tiles.keys()]) if (!ids.includes(id)) lvTeardown(id);
  if (!ids.length) {
    if (LV.ws) { const ws = LV.ws; LV.ws = null; ws.close(); }
    lvArm(false);
    return;
  }
  lvArm(true);
  if (!LV.ws) return lvConnect();
  if (LV.ws.readyState === WebSocket.OPEN && ids.join() !== LV.sent) {
    LV.sent = ids.join(); LV.ws.send(JSON.stringify({ cams: ids }));
  }
}
function lvConnect() {
  const ws = LV.ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/live/ws`);
  ws.binaryType = 'arraybuffer';
  LV.sent = '';
  ws.onopen = () => {
    LV.backoff = 1; lvUpdate();
    LV.ping = setInterval(() => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ ping: performance.now() })), 5000);
    ws.send(JSON.stringify({ ping: performance.now() }));
  };
  ws.onclose = () => {
    clearInterval(LV.ping);
    if (LV.ws !== ws) return;                          // closed on purpose
    LV.ws = null;
    for (const id of [...LV.tiles.keys()]) lvTeardown(id);
    LV.backoff = Math.min(LV.backoff * 2, 30);         // tiles fall back to snapshots meanwhile
    setTimeout(lvUpdate, LV.backoff * 1000);
  };
  ws.onmessage = e => {
    if (typeof e.data === 'string') {
      const m = JSON.parse(e.data);
      if (m.pong != null) LV.rtt = (performance.now() - m.pong) / 1000;
      else if (m.mime) lvStart(m.cam, m.mime);
      else if (m.end) lvTeardown(m.cam);
      return;
    }
    const dv = new DataView(e.data), n = dv.getUint8(0);
    const t = LV.tiles.get(new TextDecoder().decode(new Uint8Array(e.data, 1, n)));
    if (!t) return;
    t.age = dv.getFloat64(1 + n); t.rx = performance.now();
    t.pending.push(new Uint8Array(e.data, 9 + n)); t.feed();
  };
}
function lvStart(id, mime) {
  lvTeardown(id);
  const tile = $(`#live-grid .tile[data-cam="${CSS.escape(id)}"]`);
  const MS = window.ManagedMediaSource || window.MediaSource;
  if (!tile || !MS?.isTypeSupported(mime)) return;      // stays on snapshots
  const v = $('video', tile), ms = new MS();
  const t = { v, ms, tile, pending: [], sb: null, url: URL.createObjectURL(ms), age: null, rx: 0 };
  t.feed = () => {
    const sb = t.sb;
    if (!sb || sb.updating || !t.pending.length || ms.readyState !== 'open' || LV.tiles.get(id) !== t) return;
    const b = v.buffered;                                 // keep ~10 s per tile
    if (b.length && v.currentTime - b.start(0) > 10) { sb.remove(b.start(0), v.currentTime - 4); return; }
    const chunk = t.pending.shift();
    try { sb.appendBuffer(chunk); }
    catch (e) {
      if (e.name === 'QuotaExceededError' && b.length) { t.pending.unshift(chunk); sb.remove(b.start(0), Math.max(b.start(0) + 1, v.currentTime - 2)); }
      else lvTeardown(id);
    }
  };
  LV.tiles.set(id, t);
  v.disableRemotePlayback = true;                       // required by ManagedMediaSource (Safari)
  v.src = t.url;
  ms.addEventListener('sourceopen', () => {
    if (LV.tiles.get(id) !== t) return;
    try { t.sb = ms.addSourceBuffer(mime); } catch { lvTeardown(id); return; }
    t.sb.addEventListener('updateend', t.feed); t.feed();
  }, { once: true });
  v.onplaying = () => { if (LV.tiles.get(id) === t) { tile.dataset.vid = '1'; tile.classList.add('vid'); } };
  v.play().catch(() => {});
}
function lvTeardown(id) {
  const t = LV.tiles.get(id);
  if (!t) return;
  LV.tiles.delete(id);
  t.tile.dataset.vid = '0'; t.tile.classList.remove('vid');
  t.v.onplaying = null; t.v.pause(); t.v.removeAttribute('src'); t.v.load();
  URL.revokeObjectURL(t.url);
}
function lvTick() {                      // every 500 ms: stay at the live edge, and measure how far behind we are
  const now = performance.now();
  for (const t of LV.tiles.values()) {
    const v = t.v, b = v.buffered;
    if (!b.length || v.seeking) continue;
    const end = b.end(b.length - 1);
    if (end - v.currentTime > 1.5 || v.currentTime < b.start(b.length - 1)) v.currentTime = Math.max(b.start(b.length - 1), end - 0.2);
    if (t.tile.dataset.vid === '1' && t.age != null) {
      const buf = Math.max(0, end - v.currentTime), net = LV.rtt / 2;
      (t.tile._lat ||= []).push({ t: now, age: t.age + (now - t.rx) / 1000 + buf, net, buf });
    }
  }
}
function lvOverlayDelay(tile, c) {       // seconds the video is behind the live detection state
  const last = (tile._lat || []).at(-1);
  if (!last) return 0;
  return Math.max(0, Math.min(10, last.age + last.net - (c.lag?.[c.feed] ?? 0)));
}

// Live view polls single JPEGs instead of MJPEG: an endless MJPEG response holds one of the browser's
// 6 connections per host, so with 6+ cameras every other request on the page would queue forever.
// Fetched (not <img src>) so the frame's age header and the download time can be read for the latency stat.
function pumpFrames(tile) {
  const img = $('img', tile), url = `/api/cameras/${encodeURIComponent(tile.dataset.cam)}/snapshot.jpg`;
  let shown = null;
  const next = delay => setTimeout(async () => {
    if (!tile.isConnected) return;
    if (document.hidden || F.cam || tile.dataset.vid === '1') return next(1000);   // paused: full-window view, or video plays
    const t0 = performance.now();
    try {
      const r = await fetch(`${url}?t=${Date.now()}`, { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const blob = await r.blob(), t1 = performance.now();
      const age = parseFloat(r.headers.get('X-Frame-Age'));
      if (Number.isFinite(age)) (tile._lat ||= []).push({ t: t1, age, net: (t1 - t0) / 1000 });
      const src = URL.createObjectURL(blob);
      img.onload = () => { if (shown) URL.revokeObjectURL(shown); shown = src; tile.dataset.frames = '1'; next(150); };
      img.onerror = () => { URL.revokeObjectURL(src); tile.dataset.frames = '0'; next(2000); };
      img.src = src;
    } catch { tile.dataset.frames = '0'; next(2000); }
  }, delay);
  next(0);
}
// Drag tiles by their grip to reorder. Pointer events (not HTML5 DnD) so it also works on touch screens.
function startTileDrag(e, tile, grid) {
  if (e.button > 0) return;
  e.preventDefault();
  const before = $$('.tile', grid).map(t => t.dataset.cam);
  const r = tile.getBoundingClientRect();
  const ghost = tile.cloneNode(true);
  ghost.classList.add('drag-ghost');
  Object.assign(ghost.style, { width: r.width + 'px', height: r.height + 'px', left: r.left + 'px', top: r.top + 'px' });
  $('img', ghost).src = $('img', tile).src;
  document.body.appendChild(ghost);
  tile.classList.add('drag-src');
  const dx = e.clientX - r.left, dy = e.clientY - r.top;
  let scrollTimer = null;
  const move = ev => {
    ghost.style.left = ev.clientX - dx + 'px'; ghost.style.top = ev.clientY - dy + 'px';
    // auto-scroll near the window edges
    clearInterval(scrollTimer);
    const edge = ev.clientY < 60 ? -12 : ev.clientY > innerHeight - 60 ? 12 : 0;
    if (edge) scrollTimer = setInterval(() => scrollBy(0, edge), 16);
    const over = document.elementFromPoint(ev.clientX, ev.clientY)?.closest('#live-grid .tile');
    if (!over || over === tile) return;
    const o = over.getBoundingClientRect();
    const after = ev.clientX > o.left + o.width / 2;      // tiles flow left→right, so split by x
    grid.insertBefore(tile, after ? over.nextSibling : over);
  };
  const end = async () => {
    removeEventListener('pointermove', move); removeEventListener('pointerup', end); removeEventListener('pointercancel', end);
    clearInterval(scrollTimer);
    ghost.remove(); tile.classList.remove('drag-src');
    const order = $$('.tile', grid).map(t => t.dataset.cam);
    if (order.join() === before.join()) return;
    cameras.sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id)); refreshCameraSelects();
    try {
      await api('/api/cameras/order', { method: 'PUT', body: { ids: order } });
      // Setup reloads from the server unless it has unsaved edits; then reorder its copy so a later save keeps this order.
      if (S.cfg && S.dirty) {
        const pos = id => { const k = order.indexOf(id); return k < 0 ? Infinity : k; };
        S.cfg.cameras.sort((a, b) => pos(a.orig_id || a.id) - pos(b.orig_id || b.id));
        S.open = new Set(); S.probe = {};
      }
    }
    catch (err) {           // put tiles back if the server didn't take it
      before.forEach(id => grid.appendChild($(`.tile[data-cam="${CSS.escape(id)}"]`, grid)));
      console.error('reorder failed', err);
    }
  };
  addEventListener('pointermove', move); addEventListener('pointerup', end); addEventListener('pointercancel', end);
}
// ---- full-window live view: the camera's own H.264 at full resolution (+ audio), streamed as fMP4
const F = { cam: null, timers: [], retry: 0 };
function openFocus(id) {
  const c = cameras.find(x => x.id === id), v = $('#focus-video');
  F.cam = id; F.retry = 0;
  $('#focus-name').textContent = c?.name || id; $('#focus-res').textContent = '';
  $('#focus').classList.remove('hidden');
  lvUpdate();                                           // tiles stop streaming while full-window
  v.muted = false; updateFocusMute(); boostAudio(v);
  startFocusStream();
  F.timers.push(setInterval(() => {                     // stay at the live edge; hop over small gaps
    const b = v.buffered;
    if (!b.length || v.seeking) return;
    const end = b.end(b.length - 1);
    let inside = false;
    for (let i = 0; i < b.length; i++) if (v.currentTime >= b.start(i) && v.currentTime < b.end(i)) inside = true;
    if (end - v.currentTime > 1.5 || !inside) v.currentTime = Math.max(b.start(b.length - 1), end - 0.3);
  }, 500));
  F.timers.push(setInterval(async () => {               // detection boxes / zones on top
    const cv = $('#focus-canvas');
    if (!(OV.ai || OV.zones) || !v.videoWidth) { cv.getContext('2d').clearRect(0, 0, cv.width, cv.height); return; }
    const s = await api(`/api/cameras/${encodeURIComponent(id)}/state`).catch(() => null);
    if (s && F.cam === id) drawOverlay(cv, v, s, id);
  }, 500));
}
// Media Source Extensions: we fetch the endless fMP4 ourselves and feed it to the <video>. Plain
// <video src=live.mp4> can't work out the timeline of a file with no end and stalls at 0:00.
async function startFocusStream(withAudio = true) {
  stopFocusStream();
  const v = $('#focus-video'), msg = $('#focus-msg'), cam = F.cam;
  const ctrl = new AbortController(); F.abort = ctrl;
  msg.textContent = 'Connecting…'; msg.classList.remove('hidden');
  const MS = window.ManagedMediaSource || window.MediaSource;   // ManagedMediaSource: iPad/iPhone Safari 17.1+
  try {
    if (!MS) throw new Error('This browser has no Media Source support');
    const res = await fetch(`/api/cameras/${encodeURIComponent(cam)}/live.mp4?audio=${withAudio ? 1 : 0}`, { signal: ctrl.signal });
    if (!res.ok) throw new Error(res.status === 503 ? 'Camera is not streaming right now' : `HTTP ${res.status}`);
    const mime = res.headers.get('X-Mime') || 'video/mp4; codecs="avc1.42e01f"';
    if (!MS.isTypeSupported(mime)) {
      ctrl.abort();
      if (withAudio && /mp4a/.test(mime)) return startFocusStream(false);     // try video only
      throw new Error(`This browser can't play ${mime}`);
    }
    const ms = new MS(); F.ms = ms;
    v.disableRemotePlayback = true;                                         // required by ManagedMediaSource
    v.src = F.url = URL.createObjectURL(ms);
    await new Promise(r => ms.addEventListener('sourceopen', r, { once: true }));
    const sb = ms.addSourceBuffer(mime);
    const pending = [];
    const feed = () => {
      if (sb.updating || !pending.length || ms.readyState !== 'open' || F.abort !== ctrl) return;
      const b = v.buffered;                                                 // keep ~30 s in memory
      if (b.length && v.currentTime - b.start(0) > 30) { sb.remove(b.start(0), v.currentTime - 10); return; }
      const chunk = pending.shift();
      try { sb.appendBuffer(chunk); }
      catch (e) {
        if (e.name !== 'QuotaExceededError') throw e;
        pending.unshift(chunk);
        if (b.length) sb.remove(b.start(0), Math.max(b.start(0) + 1, v.currentTime - 5));
      }
    };
    sb.addEventListener('updateend', feed);
    v.play().catch(() => { v.muted = true; updateFocusMute(); v.play().catch(() => {}); });   // autoplay policy
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done || F.abort !== ctrl) break;
      pending.push(value); feed();
    }
    if (F.abort === ctrl) throw new Error('stream ended');
  } catch (e) {
    if (ctrl.signal.aborted && F.abort !== ctrl) return;                    // we closed it on purpose
    if (F.abort === ctrl) focusReconnect(e.message);
  }
}
function stopFocusStream() {
  const v = $('#focus-video');
  if (F.abort) { F.abort.abort(); F.abort = null; }
  v.pause(); v.removeAttribute('src'); v.load();
  if (F.url) { URL.revokeObjectURL(F.url); F.url = null; }
  F.ms = null;
}
function focusReconnect(why) {                           // stream ended/failed: retry a few times
  if (!F.cam) return;
  const msg = $('#focus-msg'); msg.classList.remove('hidden');
  if (++F.retry > 5) { msg.textContent = `Live stream unavailable — ${why || 'the camera may be offline'}.`; return; }
  msg.textContent = 'Reconnecting…';
  setTimeout(() => F.cam && startFocusStream(), 1500);
}
function closeFocus() {
  if (!F.cam) return;
  F.cam = null; F.timers.forEach(clearInterval); F.timers = [];
  stopFocusStream();
  $('#focus').classList.add('hidden');
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  lvUpdate();                                           // tiles resume streaming
}
function updateFocusMute() {
  const v = $('#focus-video'); $('#focus-mute').textContent = v.muted ? '🔇 Unmute' : '🔊 Mute';
}
(() => {
  const v = $('#focus-video');
  v.addEventListener('playing', () => {
    F.retry = 0; $('#focus-msg').classList.add('hidden');
    $('#focus-res').textContent = `${v.videoWidth}×${v.videoHeight}`;
  });
  v.addEventListener('error', () => { if (F.cam && v.getAttribute('src')) { stopFocusStream(); focusReconnect(v.error?.message); } });
  v.ondblclick = closeFocus;
  $('#focus-close').onclick = closeFocus;
  $('#focus-mute').onclick = () => { v.muted = !v.muted; updateFocusMute(); };
  $('#focus-fs').onclick = () => document.fullscreenElement ? document.exitFullscreen() : $('#focus').requestFullscreen().catch(() => {});
  addEventListener('keydown', e => { if (e.key === 'Escape' && F.cam && !document.fullscreenElement) closeFocus(); });
})();

function friendlyCamError(err, state) {
  if (!err) return state === 'init' ? 'Connecting…' : 'No video yet';
  if (/401|unauthori[sz]ed|authoriz/i.test(err)) return 'Login failed — check this camera\'s username/password (⚙ → Setup)';
  if (/refused/i.test(err)) return 'Camera refused the connection — is RTSP enabled on it, and the port right?';
  if (/timed? ?out|unreachable|no route/i.test(err)) return 'Can\'t reach the camera — check its IP address and that it is online';
  if (/404|not found/i.test(err)) return 'Stream path not found — check the brand/model setting';
  return err;
}
function editCamera(id) { S.pendingOpen = id; showTab('setup'); }

// ---------------------------------------------------------------- events
async function loadEvents() {
  const cam = $('#ev-cam').value;
  const evs = await api(`/api/events?limit=200${cam ? '&camera_id=' + encodeURIComponent(cam) : ''}`);
  $('#ev-list').innerHTML = evs.map(e => `
    <div class="card" data-id="${e.id}">
      ${e.snapshot_path ? `<img loading="lazy" src="/media/${esc(e.snapshot_path)}">` : '<img>'}
      <div class="body">
        <div class="title"><span>${esc(camName(e.camera_id))}</span><span class="muted">${fmtDur(e.end_ts ? e.end_ts - e.start_ts : null)}</span></div>
        <div class="muted">${fmtTime(e.start_ts)} · ${esc(e.rule_name || e.source)}</div>
        <div class="chips">${e.labels.map(l => `<span class="chip">${esc(l)}</span>`).join('')}${e.plates.map(p => `<span class="chip plate">${esc(p)}</span>`).join('')}</div>
      </div>
    </div>`).join('') || '<p class="muted">No events yet.</p>';
  const list = evs.map(e => ({ id: e.id }));             // newest first, as shown; the player's arrows walk it
  $$('#ev-list .card').forEach((c, i) => c.onclick = () => openFromList(list, i, 'events'));
}
$('#ev-refresh').onclick = loadEvents; $('#ev-cam').onchange = loadEvents;

// Player arrows walk the list the video was opened from:
//  - Events (newest first): ‹ earlier = next index, › later = previous index
//  - Search (ranked results): ‹ previous result, › next result; each opens at its matched object
const EV = { list: [], i: -1, mode: 'events' };
function openFromList(list, i, mode) {
  EV.list = list; EV.i = i; EV.mode = mode;
  openEvent(list[i].id, list[i].seek);
}
function stepEvent(arrow) {                              // arrow: -1 = ‹, +1 = ›
  const j = EV.i + (EV.mode === 'events' ? -arrow : arrow);
  if (EV.i < 0 || j < 0 || j >= EV.list.length) return;
  EV.i = j; openEvent(EV.list[j].id, EV.list[j].seek);
}
function updateEventNav() {
  const n = EV.list.length, i = EV.i, ev = EV.mode === 'events';
  const can = d => i >= 0 && (j => j >= 0 && j < n)(i + (ev ? -d : d));
  $('#pl-prev').disabled = !can(-1); $('#pl-next').disabled = !can(1);
  $('#pl-prev').title = ev ? 'Earlier event (←)' : 'Previous result (←)';
  $('#pl-next').title = ev ? 'Later event (→)' : 'Next result (→)';
  $('#pl-pos').textContent = i < 0 ? '' : ev ? `${n - i} / ${n}` : `${EV.mode === 'vehicle' ? 'Sighting' : 'Result'} ${i + 1} / ${n}`;
  $('#pl-prev').hidden = $('#pl-next').hidden = i < 0;
}
$('#pl-prev').onclick = () => stepEvent(-1);
$('#pl-next').onclick = () => stepEvent(1);
$('#player').addEventListener('keydown', e => {
  if (e.target.matches('input, textarea, select') || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();                                   // instead of the video's own 5 s seek
    $(e.key === 'ArrowLeft' ? '#pl-prev' : '#pl-next').click();
  }
});
// ---- audio boost: the cameras send audio at about -70 dBFS (peaks around -45), far too quiet to hear.
// Played audio goes through a gain and then a limiter, so loud moments don't distort. "Auto level" (0) measures
// the sound as it plays and raises it toward a normal listening level, up to x200. Browser-only: clips untouched.
const AGC = { target: -26, max: 200, floor: -85 };     // dBFS aimed for; most gain; below this it's silence
const BOOST = { ctx: null, nodes: new Map(), gain: (() => { try { const v = localStorage.getItem('theeye.audioBoost'); return v === null ? 0 : +v; } catch { return 0; } })() };
function boostAudio(v) {
  try {
    BOOST.ctx ||= new (window.AudioContext || window.webkitAudioContext)();
    if (!BOOST.nodes.has(v)) {                 // an element can only be routed once
      const ctx = BOOST.ctx, src = ctx.createMediaElementSource(v), g = ctx.createGain(), lim = ctx.createDynamicsCompressor();
      const meter = ctx.createAnalyser(); meter.fftSize = 2048;
      lim.threshold.value = -12; lim.knee.value = 6; lim.ratio.value = 20; lim.attack.value = 0.003; lim.release.value = 0.25;
      src.connect(g).connect(lim).connect(ctx.destination);
      src.connect(meter);                    // level before the gain, for auto level
      BOOST.nodes.set(v, g);
      v.addEventListener('play', () => ctx.resume().catch(() => {}));
      const buf = new Float32Array(meter.fftSize);
      let level = -66;                        // slow running level (dB); starts at these cameras' usual level,
      if (BOOST.gain === 0) g.gain.value = 10 ** ((AGC.target - level) / 20);   // so play/seek is audible at once
      setInterval(() => {
        if (BOOST.gain !== 0 || v.paused) return;
        meter.getFloatTimeDomainData(buf);
        const db = 10 * Math.log10(buf.reduce((s, x) => s + x * x, 0) / buf.length + 1e-12);
        if (db > AGC.floor) level = db > level ? level + (db - level) * 0.5 : level + (db - level) * 0.05;
        const want = Math.min(AGC.max, Math.max(1, 10 ** ((AGC.target - level) / 20)));
        g.gain.setTargetAtTime(want, ctx.currentTime, want < g.gain.value ? 0.05 : 0.8);   // down fast, up slowly
      }, 100);
    }
    if (BOOST.gain) BOOST.nodes.get(v).gain.value = BOOST.gain;
    BOOST.ctx.resume().catch(() => {});
  } catch (e) { console.warn('audio boost unavailable', e); }
}
$$('select.boost').forEach(s => {
  s.value = String(BOOST.gain);
  s.onchange = () => {
    BOOST.gain = +s.value;
    try { localStorage.setItem('theeye.audioBoost', s.value); } catch { /* not kept */ }
    $$('select.boost').forEach(o => { o.value = s.value; });
    for (const g of BOOST.nodes.values()) if (BOOST.gain) g.gain.value = BOOST.gain;
  };
});

async function openEvent(id, seek) {
  const token = EV.token = {};                            // newest request wins when stepping quickly
  updateEventNav();
  const e = await api(`/api/events/${id}`);
  if (EV.token !== token) return;
  $('#pl-title').textContent = `${camName(e.camera_id)} — ${fmtTime(e.start_ts)} — ${e.rule_name || e.source}`;
  const v = $('#pl-video');
  boostAudio(v);
  if (e.clip_path) {
    v.src = `/media/${e.clip_path}`;
    v.onloadedmetadata = () => { v.currentTime = seek ?? e.play_from ?? e.clip_offset; v.play().catch(() => {}); };
  } else { v.removeAttribute('src'); v.load(); }
  $('#pl-info').innerHTML = `
    ${e.clip_path ? `<div class="row small muted" style="gap:8px;margin-bottom:6px">
        ${e.play_from > e.clip_offset + 0.5 ? `Starts where activity begins — ${(e.play_from - e.clip_offset).toFixed(0)} s of earlier footage before it.` : ''}
        <button type="button" id="pl-pre">⏮ From pre-roll</button>
        <button type="button" id="pl-act">▶ From activity</button></div>`
      : '<p class="muted">No clip (recording disabled, pruned, or still pending).</p>'}
    <div><b>Detections</b> (click to jump)<br>${e.detections.map(d => `
      <span class="det" data-seek="${d.seek}">${d.snapshot_path ? `<img src="/media/${esc(d.snapshot_path)}">` : ''}
      ${esc(animalName(d))} ${(d.confidence * 100).toFixed(0)}%${d.plate ? ` <span class="chip plate">${esc(d.plate)}</span>` : ''}</span>`).join('') || '<span class="muted">none</span>'}</div>
    <div style="margin-top:6px"><b>Motion triggers</b>: ${e.motion.length} ${e.motion.map(m => `<span class="chip">zone ${m.zone_id || 'full'} · ${fmtDur(m.end_ts ? m.end_ts - m.start_ts : null)}</span>`).join(' ')}</div>
    <div style="margin-top:8px"><button id="pl-del" class="danger">Delete event</button></div>`;
  $$('#pl-info .det').forEach(d => d.onclick = () => { v.currentTime = +d.dataset.seek; v.play(); });
  if ($('#pl-pre')) {
    $('#pl-pre').onclick = () => { v.currentTime = e.clip_offset; v.play(); };
    $('#pl-act').onclick = () => { v.currentTime = e.play_from ?? e.clip_offset; v.play(); };
  }
  $('#pl-del').onclick = async () => { await api(`/api/events/${id}`, { method: 'DELETE' }); $('#player').close(); loadEvents(); };
  if (!$('#player').open) $('#player').showModal();
}
$('#pl-close').onclick = () => $('#player').close();
$('#player').addEventListener('close', () => $('#pl-video').pause());

// ---------------------------------------------------------------- vehicles (repeat visitors)
const VH = (() => { try { return { status: localStorage.getItem('theeye.vehFilter') || 'flagged' }; } catch { return { status: 'flagged' }; } })();
$$('#veh-filter button').forEach(b => b.onclick = () => {
  VH.status = b.dataset.s; try { localStorage.setItem('theeye.vehFilter', VH.status); } catch { /* not persisted */ }
  loadVehicles();
});
async function loadVehicles() {
  $$('#veh-filter button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.s === VH.status)));
  const r = await api(`/api/vehicles?status=${VH.status}`);
  updateVehicleBadge(r.counts);
  $('#veh-help').textContent = `A vehicle seen driving on ${r.repeat_visits} separate visits (30+ min apart) is flagged; parked cars don't count. ` +
    'Matched by licence plate when readable, otherwise by appearance, so two identical cars can be merged. Mark your own cars as Known.';
  $('#veh-list').innerHTML = r.vehicles.map(v => `
    <div class="card veh ${v.status}" data-id="${v.id}">
      ${v.snapshot_path ? `<img loading="lazy" src="/media/${esc(v.snapshot_path)}">` : '<img>'}
      <div class="body">
        <div class="title"><span>${esc(v.label)}${v.plate ? ` <span class="chip plate">${esc(v.plate)}</span>` : ''}</span>
          ${v.status === 'new' ? '' : `<span class="status">${v.status === 'flagged' ? '🚩 flagged' : 'known'}</span>`}</div>
        <div><b>${v.visits}</b> visit${v.visits === 1 ? '' : 's'} · ${v.sightings} sighting${v.sightings === 1 ? '' : 's'} · ${v.cameras.map(camName).map(esc).join(', ')}</div>
        <div class="muted">last ${fmtTime(v.last_ts)}${v.note ? ` · ${esc(v.note)}` : ''}</div>
        <div class="row2">
          ${v.status !== 'known' ? '<button type="button" data-st="known" title="Your own or a neighbour&#39;s car: never flag">Mark known</button>' : ''}
          ${v.status === 'flagged' ? '<button type="button" data-st="new" title="Clear the flag; it flags again after more visits">Unflag</button>' : ''}
          ${v.status === 'new' ? '<button type="button" data-st="flagged">🚩 Flag</button>' : ''}
          ${v.status === 'known' ? '<button type="button" data-st="new">Not known</button>' : ''}
        </div>
      </div>
    </div>`).join('') || `<p class="muted">${VH.status === 'flagged' ? 'No flagged vehicles.' : 'No vehicles yet.'}</p>`;
  $$('#veh-list .veh').forEach(card => {
    const id = +card.dataset.id;
    $$('button[data-st]', card).forEach(b => b.onclick = async e => {
      e.stopPropagation();
      await api(`/api/vehicles/${id}`, { method: 'PUT', body: { status: b.dataset.st } });
      loadVehicles();
    });
    card.onclick = async () => {                          // play its sightings; the player's arrows step through them
      const s = (await api(`/api/vehicles/${id}/sightings`)).filter(x => x.event_id && x.clip_path);
      if (!s.length) { card.querySelector('.muted').textContent = 'No recorded clips for this vehicle.'; return; }
      openFromList(s.map(x => ({ id: x.event_id, seek: x.seek })), 0, 'vehicle');
    };
  });
}
function updateVehicleBadge(counts) {
  const n = counts?.flagged || 0, b = $('#veh-badge');
  b.textContent = `🚩${n}`; b.classList.toggle('hidden', !n);
}
setInterval(() => api('/api/vehicles?status=flagged&limit=1').then(r => updateVehicleBadge(r.counts)).catch(() => {}), 60000);
api('/api/vehicles?status=flagged&limit=1').then(r => updateVehicleBadge(r.counts)).catch(() => {});

// ---------------------------------------------------------------- search
// Every kind of object found in a recording, with its best certainty; the one that matched is highlighted.
// Detector label, or the closer look at an animal: "squirrel (was bird)", "crow", "pomeranian".
const animalName = d => d.subtype || d.animal
  ? `${d.subtype || d.animal}${d.animal ? ` (detector said ${d.label})` : ''}` : d.label;
const tagChips = (tags, hit) => (tags || []).map(t =>
  `<span class="chip${t.label === hit ? ' hit' : ''}">${esc(t.label)} ${Math.round(t.conf * 100)}%</span>`).join('');
// Camera filter for searches. Stores the cameras NOT ticked, so a newly added camera is searched by default.
const SC_KEY = 'theeye.searchCamsOff';
const scOff = () => { try { return new Set(JSON.parse(localStorage.getItem(SC_KEY) || '[]')); } catch { return new Set(); } };
function renderSearchCams() {
  const off = scOff(), box = $('#search-cams');
  box.innerHTML = cameras.length ? `<label class="all"><input type="checkbox" data-all> All cameras</label>` + cameras.map(c =>
    `<label><input type="checkbox" value="${esc(c.id)}"${off.has(c.id) ? '' : ' checked'}> ${esc(c.name || c.id)}</label>`).join('') : '';
  syncSearchAll();
}
function syncSearchAll() {
  const boxes = $$('#search-cams input:not([data-all])'), all = $('#search-cams input[data-all]');
  if (!all) return;
  const n = boxes.filter(b => b.checked).length;
  all.checked = n === boxes.length; all.indeterminate = n > 0 && n < boxes.length;
}
$('#search-cams').onchange = e => {
  const boxes = $$('#search-cams input:not([data-all])');
  if (e.target.matches('[data-all]')) boxes.forEach(b => { b.checked = e.target.checked; });
  syncSearchAll();
  try { localStorage.setItem(SC_KEY, JSON.stringify(boxes.filter(b => !b.checked).map(b => b.value))); } catch { /* not kept */ }
  if ($('#search-q').value.trim()) $('#search-form').requestSubmit();     // re-run with the new cameras
};
function searchCams() {                  // null = all cameras (also covers cameras added since the page loaded)
  const boxes = $$('#search-cams input:not([data-all])');
  return boxes.every(b => b.checked) ? null : boxes.filter(b => b.checked).map(b => b.value);
}

$('#search-form').onsubmit = async ev => {
  ev.preventDefault();
  const q = $('#search-q').value.trim(); if (!q) return;
  $('#search-meta').textContent = 'Thinking…'; $('#search-results').innerHTML = '';
  try {
    const cams = searchCams();
    const r = await api('/api/search', { method: 'POST', body: cams ? { q, cameras: cams } : { q } });
    const f = r.filter;
    const parts = [];
    if (f.time_label) parts.push(`🕑 ${f.time_label}`);
    if (f.cameras?.length) parts.push(`cameras=${f.cameras.map(camName).join(', ')}`);
    if (f.labels?.length) parts.push(`labels=${f.labels.join(',')}`);
    if (f.animals?.length) parts.push(`animals=${f.animals.join(',')}`);
    if (f.plate) parts.push(`plate=${f.plate}`);
    if (f.zones?.length) parts.push(`zones=${f.zones.join(',')}`);
    if (!f.time_label) {
      if (f.start) parts.push(`from ${f.start}`); if (f.end) parts.push(`to ${f.end}`);
      if (f.hours?.length) parts.push(`hours=${f.hours.join(',')}`);
    }
    if (f.min_confidence) parts.push(`certainty ≥ ${Math.round(f.min_confidence * 100)}%`);
    if (f.visual) parts.push(`looks like “${f.visual}”`);
    $('#search-meta').innerHTML = `${esc(f.explanation || '')} — <code>${esc(r.target)}: ${esc(parts.join(' · ') || 'everything')}</code> · ${r.results.length} results · ${r.ms} ms`;
    $('#search-results').innerHTML = r.results.map(x => {
      if (x.kind === 'detection') return `
        <div class="card" data-ev="${x.event_id || ''}" data-seek="${x.seek ?? ''}">
          ${x.snapshot_path ? `<img loading="lazy" src="/media/${esc(x.snapshot_path)}">` : '<img>'}
          <div class="body"><div class="title"><span>${esc(animalName(x))} ${(x.confidence * 100).toFixed(0)}%</span><span class="muted">${x.score != null ? 'look-alike ' + x.score.toFixed(2) : ''}</span></div>
          <div class="muted">${esc(camName(x.camera_id))} · ${fmtTime(x.first_ts)}${x.event_id ? '' : ' · no clip'}</div>
          <div class="chips">${tagChips(x.tags, x.subtype || x.animal || x.label)}${x.plate ? `<span class="chip plate">${esc(x.plate)}</span>` : ''}</div></div></div>`;
      if (x.kind === 'event') return `
        <div class="card" data-ev="${x.id}">${x.snapshot_path ? `<img loading="lazy" src="/media/${esc(x.snapshot_path)}">` : '<img>'}
          <div class="body"><div class="title"><span>Event #${x.id}</span></div><div class="muted">${esc(camName(x.camera_id))} · ${fmtTime(x.start_ts)}</div>
          <div class="chips">${x.tags?.length ? tagChips(x.tags) : x.labels.map(l => `<span class="chip">${esc(l)}</span>`).join('')}${x.plates.map(p => `<span class="chip plate">${esc(p)}</span>`).join('')}</div></div></div>`;
      return `<div class="card"><div class="body"><div class="title">Motion · ${esc(camName(x.camera_id))}</div>
        <div class="muted">${fmtTime(x.start_ts)} · zone ${x.zone_id || 'full'} · ${fmtDur(x.end_ts ? x.end_ts - x.start_ts : null)}</div></div></div>`;
    }).join('') || '<p class="muted">Nothing found.</p>';
    // Results with a recording, in ranked order, so the player's arrows step through them.
    const playable = $$('#search-results .card[data-ev]').filter(c => c.dataset.ev);
    const list = playable.map(c => ({ id: +c.dataset.ev, seek: c.dataset.seek ? +c.dataset.seek : undefined }));
    playable.forEach((c, i) => c.onclick = () => openFromList(list, i, 'search'));
  } catch (e) { $('#search-meta').textContent = 'Search failed: ' + e.message; }
};

// ---------------------------------------------------------------- triggers
async function loadTriggers() {
  loadSound().catch(e => console.error('sound levels', e));
  const cam = $('#tr-cam').value;
  const rows = await api(`/api/triggers?limit=500${cam ? '&camera_id=' + encodeURIComponent(cam) : ''}`);
  $('#tr-body').innerHTML = rows.map(r => `<tr><td>${esc(camName(r.camera_id))}</td><td>${esc(r.zone_name || (r.zone_id ? '#' + r.zone_id : 'Full frame'))}</td>
    <td>${fmtTime(r.start_ts)}</td><td>${fmtDur(r.end_ts ? r.end_ts - r.start_ts : null)}</td><td>${(r.peak_area * 100).toFixed(1)}%</td></tr>`).join('');
}
$('#tr-refresh').onclick = loadTriggers; $('#tr-cam').onchange = loadTriggers;

// ---- sound levels: one small chart per camera (single series each, shared -70..0 dB scale)
const SND = (() => { try { return { hours: +localStorage.getItem('theeye.soundHours') || 6 }; } catch { return { hours: 6 }; } })();
$$('#snd-range button').forEach(b => b.onclick = () => {
  SND.hours = +b.dataset.h;
  try { localStorage.setItem('theeye.soundHours', SND.hours); } catch { /* not persisted */ }
  loadSound();
});
async function loadSound() {
  $$('#snd-range button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.h === SND.hours)));
  const camSel = $('#tr-cam').value;
  const [data, rules] = await Promise.all([api(`/api/sound?hours=${SND.hours}`), api('/api/rules').catch(() => [])]);
  const cams = cameras.filter(c => (!camSel || c.id === camSel) && (data.cameras[c.id] || c.has_audio));
  $('#snd-charts').innerHTML = cams.length ? cams.map(c => `<div class="snd-card" data-cam="${esc(c.id)}"></div>`).join('')
    : '<p class="muted small">No sound data yet — cameras with a microphone appear here within a minute.</p>';
  for (const c of cams) {
    const thr = rules.filter(r => r.enabled && (r.camera_id === '*' || r.camera_id === c.id))
      .flatMap(r => r.conditions.filter(k => k.type === 'sound').map(k => ({ db: k.min_db ?? -30, name: r.name })));
    if (c.sound_threshold_db != null) thr.push({ db: c.sound_threshold_db, name: 'Sound detection' });
    drawSoundChart($(`#snd-charts .snd-card[data-cam="${CSS.escape(c.id)}"]`), c, data.cameras[c.id] || [], thr, data);
  }
}
function drawSoundChart(card, cam, pts, thr, data) {
  const W = Math.max(280, card.clientWidth - 24), H = 130, L = 30, R = 6, T = 6, B = 18;
  const t0 = data.since, t1 = Date.now() / 1000, lo = -90, hi = 0;   // Cam v4 silence sits near -87
  const x = t => L + (t - t0) / (t1 - t0) * (W - L - R), y = db => T + (hi - Math.max(lo, Math.min(hi, db))) / (hi - lo) * (H - T - B);
  let d = '', prev = null;                                  // break the line where the camera sent nothing
  for (const [t, peak] of pts) { d += `${prev == null || t - prev > data.bucket * 2.5 ? 'M' : 'L'}${x(t).toFixed(1)},${y(peak).toFixed(1)}`; prev = t; }
  const span = t1 - t0, step = [600, 1800, 3600, 3 * 3600, 6 * 3600, 86400].find(s => span / s <= 6) || 86400;
  const xt = []; for (let t = Math.ceil(t0 / step) * step; t < t1; t += step) xt.push(t);
  const fmtX = t => { const dt = new Date(t * 1000); return step >= 86400 ? dt.toLocaleDateString([], { month: 'short', day: 'numeric' }) : dt.toLocaleTimeString([], { hour: 'numeric', minute: step < 3600 ? '2-digit' : undefined }); };
  const now = pts.length ? pts[pts.length - 1] : null;
  const loudest = [...pts].sort((a, b) => b[1] - a[1]).slice(0, 8);
  card.innerHTML = `<div class="t"><b>${esc(cam.name)}</b><span class="muted small">${cam.sound_db != null ? `now ${Math.round(cam.sound_db)} dB` : (pts.length ? '' : 'no audio')}</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(cam.name)} sound level, last ${SND.hours} hours">
      <g class="grid">${[-80, -60, -40, -20, 0].map(v => `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 4}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join('')}</g>
      ${xt.map(t => `<text x="${x(t)}" y="${H - 4}" text-anchor="middle">${esc(fmtX(t))}</text>`).join('')}
      <g class="thr">${thr.map(k => `<line x1="${L}" x2="${W - R}" y1="${y(k.db)}" y2="${y(k.db)}"/><text x="${W - R}" y="${y(k.db) - 3}" text-anchor="end">${esc(k.name)} ≥ ${k.db} dB</text>`).join('')}</g>
      <path class="ln" d="${d}"/>
      <line class="xh hidden" y1="${T}" y2="${H - B}"/><circle class="dot hidden" r="4"/>
      <rect x="${L}" y="0" width="${W - L - R}" height="${H}" fill="transparent"/>
    </svg><div class="snd-tip hidden"></div>
    <details><summary class="muted">Loudest moments</summary>${loudest.length ? `<table>${loudest.map(([t, p, a]) =>
      `<tr><td>${esc(fmtTime(t))}</td><td>peak ${p} dB</td><td class="muted">avg ${a} dB</td></tr>`).join('')}</table>` : '<p class="muted">No data yet.</p>'}</details>`;
  const svg = $('svg', card), tip = $('.snd-tip', card), xh = $('.xh', svg), dot = $('.dot', svg);
  svg.onpointermove = e => {                                // crosshair + tooltip on the nearest point
    if (!pts.length) return;
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
    const t = t0 + (px - L) / (W - L - R) * (t1 - t0);
    let best = pts[0]; for (const p of pts) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p;
    const bx = x(best[0]), by = y(best[1]);
    xh.setAttribute('x1', bx); xh.setAttribute('x2', bx); dot.setAttribute('cx', bx); dot.setAttribute('cy', by);
    xh.classList.remove('hidden'); dot.classList.remove('hidden'); tip.classList.remove('hidden');
    tip.innerHTML = `${esc(fmtTime(best[0]))}<br><b>peak ${best[1]} dB</b> <span class="muted">· avg ${best[2]} dB</span>`;
    const tx = 12 + bx * r.width / W, flip = tx > card.clientWidth * 0.6;   // keep the tooltip inside the card
    tip.style.left = `${tx}px`; tip.style.top = `${30 + by * r.height / H}px`;
    tip.style.transform = flip ? 'translate(calc(-100% - 10px), -50%)' : 'translate(10px, -50%)';
  };
  svg.onpointerleave = () => { xh.classList.add('hidden'); dot.classList.add('hidden'); tip.classList.add('hidden'); };
}

// ---------------------------------------------------------------- zones
const Z = { cam: null, zones: [], sel: null, draft: null, drag: null, active: [] };
const zc = $('#zn-canvas'), zimg = $('#zn-img');

async function initZones() {
  Z.cam = $('#zn-cam').value || cameras[0]?.id; if (!Z.cam) return;
  zimg.src = `/api/cameras/${encodeURIComponent(Z.cam)}/snapshot.jpg?t=${Date.now()}`;
  await reloadZones();
  timers.zones = setInterval(async () => {
    try { Z.active = (await api(`/api/cameras/${encodeURIComponent(Z.cam)}/state`)).motion_zones; drawZones(); } catch {}
  }, 400);
  timers.zsnap = setInterval(() => { zimg.src = `/api/cameras/${encodeURIComponent(Z.cam)}/snapshot.jpg?t=${Date.now()}`; }, 2000);
}
async function reloadZones() {
  zonesCache = await api('/api/zones');
  Z.zones = zonesCache.filter(z => z.camera_id === Z.cam);
  $('#zn-list').innerHTML = Z.zones.map(z => `<li data-id="${z.id}" class="${Z.sel?.id === z.id ? 'sel' : ''}">
    <span class="sw" style="background:${esc(z.color)}"></span>${esc(z.name)} <span class="muted small">${z.kind}${z.enabled ? '' : ' · off'}</span></li>`).join('')
    || '<li class="muted">No zones — whole frame is used for motion.</li>';
  $$('#zn-list li[data-id]').forEach(li => li.onclick = () => selectZone(Z.zones.find(z => z.id === +li.dataset.id)));
  if (Z.sel) Z.sel = Z.zones.find(z => z.id === Z.sel.id) || null;
  drawZones();
}
$('#zn-cam').onchange = () => { Z.sel = Z.draft = null; $('#zn-form').classList.add('hidden'); showTab('zones'); };
zimg.onload = drawZones;
window.addEventListener('resize', drawZones);

function selectZone(z) {
  Z.sel = z ? structuredClone(z) : null; Z.draft = null;
  const f = $('#zn-form');
  f.classList.toggle('hidden', !Z.sel);
  if (Z.sel) {
    for (const k of ['name', 'kind', 'sensitivity', 'min_area', 'min_frames', 'color']) f.elements[k].value = Z.sel[k];
    f.elements.enabled.checked = Z.sel.enabled;
    f.elements.ai.checked = Z.sel.ai ?? true;
    updateVals();
    $('#zn-delete').classList.toggle('hidden', !Z.sel.id);
  }
  $$('#zn-list li').forEach(li => li.classList.toggle('sel', +li.dataset.id === Z.sel?.id));
  drawZones();
}
function updateVals() {
  const f = $('#zn-form');
  $$('.val', f).forEach(s => {
    const v = +f.elements[s.dataset.for].value;
    s.textContent = s.dataset.for === 'min_area' ? (v * 100).toFixed(1) + '%' : v.toFixed(2);
  });
}
$('#zn-form').oninput = updateVals;

const toNorm = e => { const r = zc.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
const nearPt = (pts, p, px = 10) => { const r = zc.getBoundingClientRect(); return pts.findIndex(([x, y]) => Math.hypot((x - p[0]) * r.width, (y - p[1]) * r.height) < px); };

$('#zn-new').onclick = () => {
  const colors = ['#ff3b30', '#30d158', '#0a84ff', '#ff9f0a', '#bf5af2', '#64d2ff'];
  Z.draft = { id: null, camera_id: Z.cam, name: `Zone ${Z.zones.length + 1}`, points: [], kind: 'motion', sensitivity: 0.5,
    min_area: 0.02, min_frames: 2, color: colors[Z.zones.length % colors.length], enabled: true };
  Z.sel = null; $('#zn-form').classList.add('hidden'); drawZones();
};
zc.addEventListener('mousedown', e => {
  const p = toNorm(e);
  if (Z.draft) {
    if (Z.draft.points.length >= 3 && nearPt(Z.draft.points, p) === 0) return finishDraft();
    Z.draft.points.push(p); drawZones(); return;
  }
  if (Z.sel) {
    const i = nearPt(Z.sel.points, p);
    if (i >= 0) { Z.drag = i; return; }
  }
  // select zone by clicking inside it
  const hit = [...Z.zones].reverse().find(z => pointInPoly(p, z.points));
  if (hit) selectZone(hit);
});
zc.addEventListener('mousemove', e => {
  if (Z.drag != null && Z.sel) { Z.sel.points[Z.drag] = toNorm(e).map(v => Math.min(1, Math.max(0, v))); drawZones(); }
  else if (Z.draft) { Z.hover = toNorm(e); drawZones(); }
});
window.addEventListener('mouseup', () => { Z.drag = null; });
zc.addEventListener('dblclick', e => {   // double-click an edge of the selected zone to insert a point
  if (!Z.sel) return;
  const p = toNorm(e), pts = Z.sel.points;
  let best = -1, bd = 1e9;
  pts.forEach((a, i) => { const b = pts[(i + 1) % pts.length]; const d = segDist(p, a, b); if (d < bd) { bd = d; best = i; } });
  if (best >= 0 && bd < 0.02) { pts.splice(best + 1, 0, p); drawZones(); }
});
window.addEventListener('keydown', e => {
  if (!$('#tab-zones').classList.contains('active')) return;
  if (e.key === 'Enter' && Z.draft) finishDraft();
  if (e.key === 'Escape') { Z.draft = null; drawZones(); }
  if ((e.key === 'Backspace') && Z.draft?.points.length && document.activeElement === document.body) { Z.draft.points.pop(); drawZones(); }
});
function finishDraft() {
  if (Z.draft.points.length < 3) return;
  const z = Z.draft; Z.draft = null; selectZone(z); $('#zn-form').elements.name.focus();
}
$('#zn-form').onsubmit = async e => {
  e.preventDefault();
  const f = e.target;
  const z = { ...Z.sel, name: f.elements.name.value, kind: f.elements.kind.value, sensitivity: +f.elements.sensitivity.value,
    min_area: +f.elements.min_area.value, min_frames: +f.elements.min_frames.value, color: f.elements.color.value, enabled: f.elements.enabled.checked,
    ai: f.elements.ai.checked };
  const r = z.id ? await api(`/api/zones/${z.id}`, { method: 'PUT', body: z }) : await api('/api/zones', { method: 'POST', body: z });
  Z.sel = { ...z, id: r.id }; await reloadZones(); selectZone(Z.zones.find(x => x.id === r.id));
};
$('#zn-delete').onclick = async () => {
  if (!Z.sel?.id) return;
  await api(`/api/zones/${Z.sel.id}`, { method: 'DELETE' }); selectZone(null); reloadZones();
};

function drawZones() {
  if (!zimg.clientWidth) return;
  const w = zc.width = zimg.clientWidth, h = zc.height = zimg.clientHeight, ctx = zc.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  const poly = (pts, color, { fill, active, sel, open } = {}) => {
    if (!pts.length) return;
    ctx.beginPath(); pts.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](x * w, y * h));
    if (!open) ctx.closePath();
    ctx.strokeStyle = color; ctx.lineWidth = sel ? 3 : 2; ctx.stroke();
    ctx.fillStyle = color + (active ? '66' : fill ? '22' : '00'); if (!open) ctx.fill();
    if (sel || open) pts.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x * w, y * h, i === 0 && open ? 7 : 5, 0, 7); ctx.fillStyle = i === 0 && open ? '#fff' : color; ctx.fill(); });
  };
  for (const z of Z.zones) {
    if (Z.sel?.id === z.id) continue;
    ctx.setLineDash(z.kind === 'exclude' ? [6, 4] : []);
    poly(z.points, z.color, { fill: true, active: Z.active.includes(z.id) });
    ctx.setLineDash([]);
    ctx.fillStyle = z.color; ctx.font = '13px system-ui';
    const [x, y] = z.points[0]; ctx.fillText(z.name, x * w + 6, y * h + 16);
  }
  if (Z.sel) poly(Z.sel.points, Z.sel.color, { fill: true, sel: true, active: Z.active.includes(Z.sel.id) });
  if (Z.draft) poly(Z.hover ? [...Z.draft.points, Z.hover] : Z.draft.points, Z.draft.color, { open: true });
}
function pointInPoly([x, y], pts) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

// ---------------------------------------------------------------- rules
const R = { rules: [], sel: null };
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
$('#rl-days').innerHTML = DAYS.map((d, i) => `<label><input type="checkbox" value="${i}">${d}</label>`).join('');

async function initRules() {
  zonesCache = await api('/api/zones');
  R.rules = await api('/api/rules');
  $('#rl-form').elements.camera_id.innerHTML = '<option value="*">All cameras</option>' + cameras.map(c => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
  $('#rl-list').innerHTML = R.rules.map(r => `<li data-id="${r.id}" class="${R.sel?.id === r.id ? 'sel' : ''}">
    <div><b>${esc(r.name)}</b> ${r.enabled ? '' : '<span class="badge">disabled</span>'}<br>
    <span class="muted small">${r.camera_id === '*' ? 'All cameras' : esc(camName(r.camera_id))} · ${r.match.toUpperCase()} of: ${r.conditions.map(condText).join(', ')}</span></div></li>`).join('');
  $$('#rl-list li').forEach(li => li.onclick = () => selectRule(R.rules.find(r => r.id === +li.dataset.id)));
}
function condText(c) {
  const zone = c.zone_id ? ` in ${zonesCache.find(z => z.id === +c.zone_id)?.name || 'zone ' + c.zone_id}` : '';
  if (c.type === 'motion') return `motion${zone}`;
  if (c.type === 'object') return `${(c.labels || []).join('/') || 'object'}${zone}`;
  if (c.type === 'plate') return `plate ${c.pattern || '*'}${zone}`;
  return c.type;
}
function selectRule(r) {
  R.sel = r ? structuredClone(r) : null;
  const f = $('#rl-form'); f.classList.toggle('hidden', !r); if (!r) return;
  for (const k of ['name', 'camera_id', 'match', 'preroll', 'postroll', 'max_length']) f.elements[k].value = r[k];
  f.elements.webhook.value = r.webhook || '';
  f.elements.record.checked = r.record; f.elements.enabled.checked = r.enabled;
  const s = r.schedule || {};
  $$('#rl-days input').forEach(cb => cb.checked = s.days ? s.days.includes(+cb.value) : false);
  f.elements.s_start.value = s.start || ''; f.elements.s_end.value = s.end || '';
  renderConds();
  $('#rl-delete').classList.toggle('hidden', !r.id);
  $$('#rl-list li').forEach(li => li.classList.toggle('sel', +li.dataset.id === r.id));
}
function zoneOptions(sel) {
  const cam = $('#rl-form').elements.camera_id.value;
  const zs = zonesCache.filter(z => cam === '*' || z.camera_id === cam);
  return `<option value="">any zone</option>` + zs.map(z => `<option value="${z.id}" ${+sel === z.id ? 'selected' : ''}>${esc(camName(z.camera_id))}: ${esc(z.name)}</option>`).join('');
}
function renderConds() {
  $('#rl-conds').innerHTML = R.sel.conditions.map((c, i) => {
    let body = '';
    if (c.type === 'object') body = `<div class="labels">${LABELS.map(l => `<label><input type="checkbox" data-k="labels" value="${l}" ${(c.labels || []).includes(l) ? 'checked' : ''}>${l}</label>`).join('')}</div>
      min conf <input type="number" step="0.05" min="0" max="1" data-k="min_confidence" value="${c.min_confidence ?? 0.5}" style="width:70px">
      count ≥ <input type="number" min="1" data-k="min_count" value="${c.min_count ?? 1}" style="width:60px">`;
    if (c.type === 'plate') body = `<input type="text" data-k="pattern" placeholder="ABC* , *123 (glob)" value="${esc(c.pattern || '')}">`;
    if (c.type === 'sound') body = `louder than <input type="number" step="1" min="-80" max="0" data-k="min_db" value="${c.min_db ?? -30}" style="width:70px"> dB
      for <input type="number" step="0.1" min="0.1" max="60" data-k="min_seconds" value="${c.min_seconds ?? 0.5}" style="width:64px"> s
      <span class="muted small">— see Motion log → Sound levels for each camera's normal level</span>`;
    return `<div class="cond" data-i="${i}"><b>${c.type}</b>${body}${c.type === 'sound' ? '' : `<select data-k="zone_id">${zoneOptions(c.zone_id)}</select>`}
      <button type="button" class="danger" data-del>✕</button></div>`;
  }).join('') || '<p class="muted small">Add at least one condition.</p>';
  $$('#rl-conds .cond').forEach(el => {
    const c = R.sel.conditions[+el.dataset.i];
    el.oninput = el.onchange = () => {
      if (c.type === 'object') { c.labels = $$('input[data-k=labels]:checked', el).map(x => x.value); c.min_confidence = +$('[data-k=min_confidence]', el).value; c.min_count = +$('[data-k=min_count]', el).value; }
      if (c.type === 'plate') c.pattern = $('[data-k=pattern]', el).value.toUpperCase();
      if (c.type === 'sound') { c.min_db = +$('[data-k=min_db]', el).value; c.min_seconds = +$('[data-k=min_seconds]', el).value; return; }
      const z = $('[data-k=zone_id]', el).value; c.zone_id = z ? +z : null;
    };
    $('[data-del]', el).onclick = () => { R.sel.conditions.splice(+el.dataset.i, 1); renderConds(); };
  });
}
$$('[data-add]').forEach(b => b.onclick = () => {
  const t = b.dataset.add;
  R.sel.conditions.push(t === 'object' ? { type: t, labels: ['person'], zone_id: null, min_confidence: 0.5, min_count: 1 }
    : t === 'plate' ? { type: t, pattern: '*', zone_id: null }
    : t === 'sound' ? { type: t, min_db: -30, min_seconds: 0.5 } : { type: t, zone_id: null });
  renderConds();
});
$('#rl-form').elements.camera_id.onchange = renderConds;
$('#rl-new').onclick = () => selectRule({ id: null, name: 'New rule', camera_id: '*', enabled: true, match: 'any', conditions: [{ type: 'motion', zone_id: null }],
  preroll: 10, postroll: 10, max_length: 600, schedule: null, record: true, webhook: null });
$('#rl-form').onsubmit = async e => {
  e.preventDefault();
  const f = e.target;
  const days = $$('#rl-days input:checked').map(x => +x.value);
  const sched = (days.length || f.elements.s_start.value) ? { days: days.length ? days : null, start: f.elements.s_start.value || null, end: f.elements.s_end.value || null } : null;
  const body = { name: f.elements.name.value, camera_id: f.elements.camera_id.value, match: f.elements.match.value, conditions: R.sel.conditions,
    preroll: +f.elements.preroll.value, postroll: +f.elements.postroll.value, max_length: +f.elements.max_length.value, schedule: sched,
    record: f.elements.record.checked, enabled: f.elements.enabled.checked, webhook: f.elements.webhook.value || null };
  try {
    const r = R.sel.id ? await api(`/api/rules/${R.sel.id}`, { method: 'PUT', body }) : await api('/api/rules', { method: 'POST', body });
    await initRules(); selectRule(R.rules.find(x => x.id === r.id));
  } catch (err) { alert(err.message); }
};
$('#rl-delete').onclick = async () => { if (!R.sel?.id) return; await api(`/api/rules/${R.sel.id}`, { method: 'DELETE' }); R.sel = null; $('#rl-form').classList.add('hidden'); initRules(); };

boot();

// ---------------------------------------------------------------- setup (configurator)
const ALL_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow',
  'bear', 'backpack', 'handbag', 'suitcase', 'umbrella', 'boat', 'train', 'airplane'];
const S = { cfg: null, saved: null, brands: {}, system: {}, open: new Set(), probe: {}, dirty: false, path: '' };
const getK = (o, k) => k.split('.').reduce((a, p) => a?.[p], o);
const setK = (o, k, v) => { const ps = k.split('.'); const last = ps.pop(); ps.reduce((a, p) => a[p], o)[last] = v; };
const slug = s => (s || '').toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40) || 'cam';
const fmtBytes = b => b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : (b / 1e6).toFixed(0) + ' MB';

async function initSetup(force = false) {
  if (S.cfg && S.dirty && !force) { openPending(); renderSetup(); return; }
  let r;
  try { r = await api('/api/config'); }
  catch (e) {
    banner('err', `Couldn't load the configuration (${esc(e.message)}). The server may be restarting.`,
      '<button type="button" id="cfg-retry">Retry</button>');
    $('#cfg-retry').onclick = () => initSetup(true);
    return;
  }
  S.cfg = r.config; S.saved = JSON.stringify(r.config); S.brands = r.brands; S.system = r.system; S.path = r.path;
  S.cfg.cameras.forEach(c => c.orig_id = c.id);
  (S.cfg.credentials ||= []).forEach(x => x.orig_name = x.name);
  S.saved = JSON.stringify(S.cfg);
  if (!S.cfg.cameras.length) S.open.add(0);
  openPending();
  if (!r.exists) banner('info', `No configuration file yet — fill in the form and save to create <code>${esc(r.path)}</code>.`);
  renderSetup(); loadLlmModels(); loadStorage(); loadAccount();
}
function openPending() {
  if (!S.pendingOpen) return;
  const i = S.cfg.cameras.findIndex(c => c.id === S.pendingOpen);
  S.pendingOpen = null;
  if (i < 0) return;
  S.open = new Set([i]);
  setTimeout(() => $(`#cfg-cams .cam[data-i="${i}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
}
function markDirty() {
  S.dirty = JSON.stringify(S.cfg) !== S.saved;
  $('#cfg-savebar').classList.toggle('dirty', S.dirty);
  $('#cfg-dirty').textContent = S.dirty ? '● Unsaved changes' : 'No changes';
  renderRam();
}
window.addEventListener('beforeunload', e => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });
function banner(kind, html, action) {
  $('#cfg-banner').innerHTML = html ? `<div class="banner ${kind}"><div>${html}</div>${action || ''}</div>` : '';
}

function renderSetup() {
  $('#cfg-path').textContent = `Config file: ${S.path}`;
  // simple fields bound by data-k
  $$('#tab-setup [data-k]').forEach(el => {
    const v = getK(S.cfg, el.dataset.k);
    const csv = el.hasAttribute('data-csv');               // a list edited as "a, b, c"
    if (el.type === 'checkbox') el.checked = !!v; else el.value = csv ? (v || []).join(', ') : v ?? '';
    el.oninput = el.onchange = () => {
      setK(S.cfg, el.dataset.k, el.type === 'checkbox' ? el.checked : el.hasAttribute('data-num') ? +el.value
        : csv ? el.value.split(',').map(s => s.trim()).filter(Boolean) : el.value);
      showVals(); markDirty();
    };
  });
  $('#cfg-classes').innerHTML = ALL_CLASSES.map(c => `<label><input type="checkbox" value="${c}" ${S.cfg.detector.classes.includes(c) ? 'checked' : ''}>${c}</label>`).join('');
  $$('#cfg-classes input').forEach(cb => cb.onchange = () => {
    S.cfg.detector.classes = $$('#cfg-classes input:checked').map(x => x.value); markDirty();
  });
  showVals(); renderCreds(); renderCams(); markDirty();
}
function showVals() { $$('#tab-setup [data-show]').forEach(s => s.textContent = (+getK(S.cfg, s.dataset.show)).toFixed(2)); }

function camSummary(c, i) {
  const p = S.probe[i]?.main;
  const where = c.brand === 'custom' ? (c.main_url || 'no URL') : `${S.brands[c.brand]?.label.split(' ')[0] || c.brand} · ${c.host || 'no host'}`;
  const st = !p ? '' : p.pending ? '<span class="muted">testing…</span>' : p.ok ? `<span class="good">✓ ${p.width}×${p.height} ${p.codec}</span>` : '<span class="warn">✗ test failed</span>';
  return `${p?.snapshot ? `<img src="${p.snapshot}">` : ''}<span class="t">${esc(c.name || c.id || 'New camera')}</span>
    <span class="muted small">${esc(c.id)} · ${esc(where)}</span>${c.enabled ? '' : '<span class="badge">disabled</span>'} ${st}
    <button type="button" class="cam-edit${S.open.has(i) ? '' : ' primary'}">${S.open.has(i) ? 'Close ▴' : 'Edit ▾'}</button>`;
}

function renderCams() {
  const box = $('#cfg-cams');
  if (!S.cfg.cameras.length) { box.innerHTML = '<p class="muted">No cameras yet.</p>'; return; }
  const hint = S.cfg.cameras.length && !S.open.size ? '<p class="small muted">Click a camera (or its <b>Edit</b> button) to change its settings.</p>' : '';
  box.innerHTML = hint + S.cfg.cameras.map((c, i) => `<div class="cam" data-i="${i}"><div class="cam-row">${camSummary(c, i)}</div>
    ${S.open.has(i) ? camEditor(c, i) : ''}</div>`).join('');
  $$('#cfg-cams .cam').forEach(el => {
    const i = +el.dataset.i, c = S.cfg.cameras[i];
    $('.cam-row', el).onclick = () => { S.open.has(i) ? S.open.delete(i) : S.open.add(i); renderCams(); };
    if (!S.open.has(i)) return;
    $$('[data-ck]', el).forEach(inp => {
      inp.oninput = inp.onchange = () => {
        const k = inp.dataset.ck;
        let v = inp.type === 'checkbox' ? inp.checked : inp.hasAttribute('data-num') ? (inp.value === '' ? null : +inp.value) : inp.value;
        if (k === 'name' && !c.orig_id && (!c.id || c.id === slug(c.name))) { c.id = slug(v); $('[data-ck=id]', el).value = c.id; }
        if (k === 'id') v = v.toLowerCase();
        c[k] = v;
        if (k === 'credential') { c.credential = v || null; renderCreds(); renderCams(); markDirty(); return; }
        if (k === 'brand') { c.port = S.brands[v]?.port ?? null; delete S.probe[i]; renderCams(); markDirty(); return; }
        if (['ring_seconds'].includes(k)) $('.ramest', el).innerHTML = ramEst(c, i);
        $('.cam-row', el).innerHTML = camSummary(c, i);
        markDirty();
      };
    });
    $$('[data-test]', el).forEach(b => b.onclick = () => testStream(i, b.dataset.test));
    const sc = $('[data-savecred]', el);
    if (sc) sc.onclick = () => saveAsLogin(c, sc);
    $('[data-dup]', el).onclick = () => duplicateCamera(i);
    $('[data-remove]', el).onclick = () => {
      if (!confirmInline(el, c)) return;
    };
  });
}
function duplicateCamera(i) {
  const src = S.cfg.cameras[i], copy = JSON.parse(JSON.stringify(src));
  delete copy.orig_id;
  let name = `${src.name || src.id} copy`, n = 2;
  while (S.cfg.cameras.some(c => c.name === name)) name = `${src.name || src.id} copy ${n++}`;
  let id = slug(name).slice(0, 40), k = 2;
  while (S.cfg.cameras.some(c => c.id === id)) id = `${slug(name).slice(0, 36)}_${k++}`;
  Object.assign(copy, { name, id });
  // Stored passwords never reach the browser; for a saved camera the server copies them from the original.
  if (src.orig_id && !copy.password) copy.from_camera = src.orig_id;
  S.cfg.cameras.splice(i + 1, 0, copy);
  const probe = {};                                   // test results are keyed by position: shift those below
  for (const [k2, v] of Object.entries(S.probe)) probe[+k2 > i ? +k2 + 1 : k2] = v;
  S.probe = probe;
  S.open = new Set([i + 1]);
  renderCreds(); renderCams(); markDirty();
  const ed = $(`#cfg-cams .cam[data-i="${i + 1}"]`);
  ed?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  const host = $('[data-ck=host]', ed), nm = $('[data-ck=name]', ed);
  (host || nm)?.focus(); (host || nm)?.select();
}
function confirmInline(el, c) {
  const b = $('[data-remove]', el);
  if (b.dataset.armed) {
    const i = S.cfg.cameras.indexOf(c);
    S.cfg.cameras.splice(i, 1); S.open = new Set(); S.probe = {}; renderCams(); markDirty(); return true;
  }
  b.dataset.armed = 1; b.textContent = c.orig_id ? 'Click again — existing zones/events stay in the DB' : 'Click again to remove';
  setTimeout(() => { if (b.isConnected) { delete b.dataset.armed; b.textContent = 'Remove camera'; } }, 4000);
  return false;
}

function camEditor(c, i) {
  const b = S.brands[c.brand] || {};
  const brandOpts = Object.entries(S.brands).map(([k, v]) => `<option value="${k}" ${k === c.brand ? 'selected' : ''}>${esc(v.label)}</option>`).join('');
  const conn = c.brand === 'custom' ? `
    <div class="fgrid">
      <label style="grid-column:1/-1">Main stream URL (high resolution — recorded)
        <input data-ck="main_url" value="${esc(c.main_url || '')}" placeholder="rtsp://user:pass@192.168.1.50:554/stream1" required></label>
      <label style="grid-column:1/-1">Sub stream URL (low resolution — used for motion &amp; AI; strongly recommended)
        <input data-ck="sub_url" value="${esc(c.sub_url || '')}" placeholder="rtsp://user:pass@192.168.1.50:554/stream2"></label>
    </div>
    <p class="small muted">Saved passwords are shown as ******** — leave them as-is to keep the stored password.</p>` : `
    <div class="fgrid">
      <label>IP address / hostname <input data-ck="host" value="${esc(c.host || '')}" placeholder="192.168.1.50" required></label>
      <label>${b.scheme === 'rtsps' ? 'RTSPS' : 'RTSP'} port <input type="number" data-ck="port" data-num value="${c.port ?? ''}" placeholder="${b.port}"></label>
      <label>Channel <input type="number" min="1" data-ck="channel" data-num value="${c.channel ?? 1}">
        <span class="small muted">1 for standalone cameras; NVR/DVR channel otherwise</span></label>
    </div>
    ${loginFields(c)}`;
  const pr = S.probe[i] || {};
  return `<div class="cam-ed">
    <div class="fgrid">
      <label>Name <input data-ck="name" value="${esc(c.name || '')}" placeholder="Driveway" required></label>
      <label>ID <input data-ck="id" value="${esc(c.id || '')}" pattern="[a-z0-9][a-z0-9_\\-]{0,39}" ${c.orig_id ? 'readonly title="ID is fixed once saved (zones, rules and events refer to it)"' : ''} required>
        <span class="small muted">${c.orig_id ? 'fixed — referenced by zones, rules, events' : 'lowercase letters, digits, _ -'}</span></label>
      <label>Brand / connection <select data-ck="brand">${brandOpts}</select></label>
      <label class="row"><input type="checkbox" data-ck="enabled" ${c.enabled ? 'checked' : ''}> Enabled</label>
    </div>
    ${b.note ? `<p class="small note">ℹ ${esc(b.note)}</p>` : ''}
    ${conn}
    <div class="row"><button type="button" data-test="main">Test main stream</button>
      <button type="button" data-test="sub" ${!b.has_sub || (c.brand === 'custom' && !c.sub_url) ? 'disabled title="This camera has no sub stream"' : ''}>Test sub stream</button>
      <span class="small muted">Connects for ~4 s: checks resolution, bitrate and keyframe interval.</span></div>
    ${['main', 'sub'].map(s => pr[s] ? probeHtml(s, pr[s], c) : '').join('')}
    <div class="sub-h">Recording &amp; analysis</div>
    <div class="fgrid">
      <label>RAM loop length (seconds) <input type="number" min="10" max="7200" step="10" data-ck="ring_seconds" data-num value="${c.ring_seconds}">
        <span class="small ramest">${ramEst(c, i)}</span></label>
      <label>Analysis frame rate (fps) <input type="number" min="1" max="30" step="1" data-ck="analyze_fps" data-num value="${c.analyze_fps}">
        <span class="small muted">frames/s checked for motion</span></label>
      <label>AI frame rate (fps) <input type="number" min="0.5" max="15" step="0.5" data-ck="detect_fps" data-num value="${c.detect_fps}">
        <span class="small muted">frames/s sent to the detector while active</span></label>
      <label class="row"><input type="checkbox" data-ck="always_detect" ${c.always_detect ? 'checked' : ''}> Run AI continuously (not only on motion)</label>
      <label class="row"><input type="checkbox" data-ck="record_audio" ${c.record_audio !== false ? 'checked' : ''}> Record audio
        <span class="small muted">— check local laws; some places require consent to record conversations</span></label>
    </div>
    <div class="sub-h">Sound detection</div>
    <div class="fgrid">
      <label class="row"><input type="checkbox" data-ck="sound_detect" ${c.sound_detect ? 'checked' : ''}> Start an event (and recording) on loud sound</label>
      <label>Volume threshold (dB) <input type="number" min="-90" max="0" step="1" data-ck="sound_threshold_db" data-num value="${c.sound_threshold_db ?? -30}">
        <span class="small muted">${soundHint(c)}</span></label>
      <label>Loud for at least (seconds) <input type="number" min="0.1" max="60" step="0.1" data-ck="sound_min_seconds" data-num value="${c.sound_min_seconds ?? 0.5}">
        <span class="small muted">ignores single clicks and pops</span></label>
    </div>
    <div class="row"><button type="button" data-dup title="Copy this camera's settings into a new camera">⧉ Duplicate</button>
      <button type="button" class="danger" data-remove>Remove camera</button></div>
  </div>`;
}

function soundHint(c) {
  const live = cameras.find(x => x.id === (c.orig_id || c.id));
  if (live && !live.has_audio) return '<span class="warn">this camera isn\'t sending audio</span>';
  const now = live?.sound_db != null ? `right now ${Math.round(live.sound_db)} dB. ` : '';
  return `0 = loudest the mic can record; quieter is more negative. ${now}Pick a level above the camera's normal background (Motion log → Sound levels).`;
}

function loginFields(c) {
  const creds = S.cfg.credentials || [];
  const cred = creds.find(x => x.name === c.credential);
  const opts = '<option value="">Enter manually</option>' + creds.map(x =>
    `<option value="${esc(x.name)}" ${x.name === c.credential ? 'selected' : ''}>${esc(x.name)} (${esc(x.username || 'no username')})</option>`).join('');
  const pick = creds.length ? `<label>Login <select data-ck="credential">${opts}</select></label>` : '';
  if (cred) return `<div class="fgrid">${pick}
    <div class="small muted login-info">Using saved login <b>${esc(cred.name)}</b> — username <b>${esc(cred.username || '—')}</b>,
      ${cred.password || cred.has_password ? 'password stored' : '<span class="warn">no password</span>'}. Edit it under <b>Saved logins</b>.</div></div>`;
  return `<div class="fgrid">${pick}
    <label>Username <input data-ck="username" value="${esc(c.username || '')}" placeholder="admin" autocomplete="off"></label>
    <label>Password <input type="password" data-ck="password" value="${esc(c.password || '')}" autocomplete="new-password"
      placeholder="${c.has_password ? '•••••• stored — leave blank to keep' : ''}"></label>
    <div class="login-info"><button type="button" data-savecred>Save as reusable login</button></div></div>`;
}
function saveAsLogin(c, btn) {
  if (!c.username) { btn.textContent = 'Enter a username first'; setTimeout(() => btn.isConnected && (btn.textContent = 'Save as reusable login'), 2500); return; }
  const base = `${(S.brands[c.brand]?.label || c.brand).split(/[ /]/)[0]} ${c.username}`;
  let name = base, n = 2;
  while (S.cfg.credentials.some(x => x.name === name)) name = `${base} ${n++}`;
  // An existing camera's stored password never reaches the browser; the server copies it via from_camera.
  S.cfg.credentials.push({ name, username: c.username, password: c.password || '',
    has_password: !!c.password || !!c.has_password, ...(!c.password && c.orig_id ? { from_camera: c.orig_id } : {}) });
  c.credential = name;
  renderCreds(); renderCams(); markDirty();
}
function renderCreds() {
  const list = S.cfg.credentials ||= [], box = $('#cfg-creds');
  if (!list.length) { box.innerHTML = '<p class="muted small">No saved logins yet — add one here, or use <b>Save as reusable login</b> on a camera.</p>'; return; }
  box.innerHTML = list.map((x, i) => {
    const used = S.cfg.cameras.filter(c => c.credential === x.name);
    return `<div class="fgrid cred" data-i="${i}">
      <label>Name <input data-cr="name" value="${esc(x.name)}" maxlength="60" required></label>
      <label>Username <input data-cr="username" value="${esc(x.username || '')}" autocomplete="off"></label>
      <label>Password <input type="password" data-cr="password" value="${esc(x.password || '')}" autocomplete="new-password"
        placeholder="${x.has_password ? '•••••• stored — leave blank to keep' : ''}"></label>
      <div class="login-info row"><span class="small muted">${used.length ? 'Used by ' + used.map(c => esc(c.name || c.id)).join(', ') : 'Not used by any camera'}</span>
        <button type="button" class="danger" data-crdel ${used.length ? 'disabled title="Switch those cameras to another login first"' : ''}>Remove</button></div>
    </div>`;
  }).join('');
  $$('#cfg-creds .cred').forEach(el => {
    const x = list[+el.dataset.i];
    $$('[data-cr]', el).forEach(inp => inp.oninput = () => {
      const k = inp.dataset.cr;
      if (k === 'name') {
        const v = inp.value.trim();
        const clash = list.some(o => o !== x && o.name === v);
        inp.setCustomValidity(clash ? 'Another saved login has this name' : '');
        S.cfg.cameras.forEach(c => { if (c.credential === x.name) c.credential = v; });
        x.name = v;
      } else x[k] = inp.value;
      renderCams(); markDirty();
    });
    $('[data-crdel]', el).onclick = () => { list.splice(+el.dataset.i, 1); renderCreds(); renderCams(); markDirty(); };
  });
}
$('#cfg-add-cred').onclick = () => {
  let n = S.cfg.credentials.length + 1;
  while (S.cfg.credentials.some(x => x.name === `Login ${n}`)) n++;
  S.cfg.credentials.push({ name: `Login ${n}`, username: '', password: '', has_password: false });
  renderCreds(); markDirty();
  $('#cfg-creds .cred:last-child [data-cr=username]')?.focus();
};

function probeHtml(stream, p, c) {
  if (p.pending) return `<div class="probe"><div class="muted">Testing ${stream} stream…</div></div>`;
  if (!p.ok) return `<div class="probe"><div class="err"><b>${stream} stream failed:</b> ${esc(p.error)}<br><span class="url">${esc(p.url || '')}</span>
    <br><span class="muted">Check IP, port, credentials, and that RTSP is enabled on the camera.</span></div></div>`;
  const warns = [];
  if (p.gop_seconds && p.gop_seconds > 2.5) warns.push(`Keyframe interval is ${p.gop_seconds}s — set it to 1–2 s on the camera (I-frame interval = frame rate) for tighter pre-roll and fresher plate reads.`);
  if (stream === 'sub' && p.width > 1280) warns.push('Sub stream is high-resolution; analysis will cost more CPU. 640×360–1280×720 is ideal.');
  if (stream === 'main' && ((c.brand === 'custom' && !c.sub_url) || !S.brands[c.brand]?.has_sub)) warns.push('No sub stream: the main stream will be decoded for analysis (much more CPU).');
  if (p.codec === 'hevc') warns.push('H.265: recorded fine, but browser playback needs HEVC support (Chrome/Edge with hardware decode, Safari).');
  const loopMb = p.bitrate_kbps ? p.bitrate_kbps * 1000 / 8 * c.ring_seconds : null;
  return `<div class="probe">${p.snapshot ? `<img src="${p.snapshot}">` : '<div></div>'}
    <div><b class="good">✓ ${stream} stream OK</b> <span class="url">${esc(p.url)}</span>
    <dl><dt>Video</dt><dd>${esc(p.codec)} ${p.width}×${p.height} @ ${p.fps ?? '?'} fps</dd>
      <dt>Bitrate</dt><dd>${p.bitrate_kbps ? (p.bitrate_kbps / 1000).toFixed(1) + ' Mbit/s' : '?'}${stream === 'main' && loopMb ? ` → ${fmtBytes(loopMb)} for a ${c.ring_seconds}s loop` : ''}</dd>
      <dt>Keyframes</dt><dd>${p.gop_seconds ? 'every ' + p.gop_seconds + ' s' : '?'}</dd>
      <dt>Audio</dt><dd>${esc(p.audio || 'none')} <span class="muted">(not recorded)</span></dd>
      <dt>Connect</dt><dd>${p.connect_ms} ms</dd></dl>
    ${warns.map(w => `<div class="warn">⚠ ${esc(w)}</div>`).join('')}</div></div>`;
}

async function testStream(i, stream) {
  const c = S.cfg.cameras[i];
  S.probe[i] = { ...(S.probe[i] || {}), [stream]: { pending: true } }; renderCams();
  try {
    S.probe[i][stream] = await api('/api/config/probe', { method: 'POST', body: { camera: c, stream, credentials: S.cfg.credentials } });
  } catch (e) { S.probe[i][stream] = { ok: false, error: e.message }; }
  renderCams(); renderRam();
}

const camBytes = (c, i) => (S.probe[i]?.main?.bitrate_kbps || 6000) * 1000 / 8 * c.ring_seconds;
function ramEst(c, i) {
  const measured = !!S.probe[i]?.main?.bitrate_kbps;
  return `<span class="muted">≈ ${fmtBytes(camBytes(c, i))} of RAM ${measured ? '(measured bitrate)' : '(assuming 6 Mbit/s — test to measure)'}</span>`;
}
function renderRam() {
  if (!S.cfg) return;
  const cams = S.cfg.cameras.map((c, i) => [c, i]).filter(([c]) => c.enabled);
  const need = cams.reduce((a, [c, i]) => a + camBytes(c, i), 0);
  const budget = S.cfg.storage.ram_budget_gb * 1e9, sys = S.system.ram_bytes || 0;
  const pct = x => Math.min(100, x / (sys || budget) * 100).toFixed(1);
  const over = need > budget;
  $('#cfg-ram').innerHTML = `<div class="rambar"><span style="width:${pct(Math.min(need, budget))}%;background:var(--accent)"></span>
      <span style="width:${pct(Math.max(0, budget - need))}%;background:#3d8bfd44"></span></div>
    <div class="small">${cams.length} camera loop(s) need <b>≈ ${fmtBytes(need)}</b> · budget ${fmtBytes(budget)}${sys ? ` · system RAM ${fmtBytes(sys)}` : ''}
    ${over ? '<span class="warn"> — over budget: oldest footage will be trimmed early, so loops will be shorter than configured</span>' : ''}
    ${sys && budget > sys * 0.75 ? '<span class="warn"> — budget is above 75% of system RAM</span>' : ''}</div>`;
}

// ---- disk usage per camera (Setup → Storage)
async function loadStorage(refresh = false) {
  const box = $('#cfg-storage');
  box.innerHTML = '<p class="muted small">Measuring…</p>';
  let r;
  try { r = await api(`/api/storage${refresh ? '?refresh=1' : ''}`); }
  catch (e) { box.innerHTML = `<p class="warn small">Couldn't measure storage: ${esc(e.message)}</p>`; return; }
  const max = Math.max(1, ...r.cameras.map(c => c.total_bytes));
  const days = t => t ? `${Math.max(0, (Date.now() / 1000 - t) / 86400).toFixed(1)} days` : '—';
  const lim = r.clip_limit_bytes, used = r.clips_bytes;
  const n = v => Number(v).toLocaleString();
  box.innerHTML = `
    <div class="sub-h">Per camera</div>
    <div class="tscroll"><table class="table stor">
      <thead><tr><th>Camera</th><th class="num">Clips</th><th class="num">Snapshots</th><th class="num">Total</th><th></th><th class="num">Oldest clip</th><th class="num">RAM loop now</th></tr></thead>
      <tbody>${r.cameras.map(c => `<tr>
        <td>${esc(c.name)}${c.configured ? '' : ' <span class="badge">removed</span>'}</td>
        <td class="num">${fmtBytes(c.clips_bytes)} <span class="muted">· ${c.clips}</span></td>
        <td class="num">${fmtBytes(c.snaps_bytes)} <span class="muted">· ${c.snaps}</span></td>
        <td class="num"><b>${fmtBytes(c.total_bytes)}</b></td>
        <td class="bar"><i style="width:${(c.total_bytes / max * 100).toFixed(1)}%"></i></td>
        <td class="num">${days(c.oldest_clip)}</td>
        <td class="num">${c.ram_bytes ? fmtBytes(c.ram_bytes) : '—'}</td></tr>`).join('')}
        <tr class="muted"><td>Database</td><td></td><td></td><td class="num">${fmtBytes(r.database_bytes)}</td><td></td><td></td><td></td></tr>
      </tbody>
      <tfoot><tr><th>Total</th><th class="num">${fmtBytes(r.clips_bytes)}</th><th class="num">${fmtBytes(r.cameras.reduce((a, c) => a + c.snaps_bytes, 0))}</th>
        <th class="num">${fmtBytes(r.total_bytes)}</th><th></th><th></th><th class="num">${fmtBytes(r.cameras.reduce((a, c) => a + c.ram_bytes, 0))}</th></tr></tfoot>
    </table></div>
    <div class="rambar" style="margin-top:14px" title="Clips vs the clip limit"><span style="width:${Math.min(100, used / lim * 100).toFixed(1)}%;background:${used > lim * 0.9 ? '#ffd60a' : 'var(--accent)'}"></span></div>
    <div class="small">Clips use <b>${fmtBytes(used)}</b> of the ${fmtBytes(lim)} limit${used > lim * 0.9 ? ' <span class="warn">— nearly full: oldest clips are being deleted</span>' : ''}
      · everything in the data folder <b>${fmtBytes(r.total_bytes)}</b> · disk free ${fmtBytes(r.disk_free)} of ${fmtBytes(r.disk_total)}</div>
    <div class="sub-h">What grows, and what keeps it in check</div>
    <div class="tscroll"><table class="table grows">
      <thead><tr><th>Data</th><th class="num">Size</th><th>Kept</th></tr></thead>
      <tbody>
        <tr><td>Video clips</td><td class="num">${fmtBytes(r.clips_bytes)} <span class="muted">· ${n(r.clips)}</span></td>
          <td>Oldest deleted above the ${fmtBytes(lim)} limit${r.oldest_clip ? ` · now go back ${days(r.oldest_clip)}` : ''}</td></tr>
        <tr><td>Snapshots</td><td class="num">${fmtBytes(r.snaps_bytes)} <span class="muted">· ${n(r.snaps)}</span></td>
          <td>As far back as the clips; older ones are deleted with them (each vehicle keeps its picture)${r.oldest_snap ? ` · now go back ${days(r.oldest_snap)}` : ''}</td></tr>
        <tr><td>Database</td><td class="num">${fmtBytes(r.database_bytes)}</td>
          <td>${n(r.rows.events)} events and ${n(r.rows.detections)} detections (${n(r.rows.searchable)} searchable by look).
            Past the clips: search data and bare detections removed, event, plate and vehicle history kept ·
            motion log ${r.keep_days.motion} days · sound levels ${r.keep_days.sound} days</td></tr>
        <tr><td>Log files</td><td class="num">${fmtBytes(r.logs_bytes)} <span class="muted">· ${r.logs}</span></td>
          <td>Rotated at ${fmtBytes(r.log_limit_bytes)}, one previous file kept</td></tr>
        ${r.other_bytes >= 1e6 ? `<tr><td>Other files</td><td class="num">${fmtBytes(r.other_bytes)}</td><td class="muted">In the data folder; not managed</td></tr>` : ''}
        <tr><td>RAM loops (memory)</td><td class="num">${fmtBytes(r.ram_bytes)}</td>
          <td>Fixed: capped by the ${fmtBytes(r.ram_budget_bytes)} RAM budget; nothing is written to disk until an event</td></tr>
      </tbody>
    </table></div>
    <p class="small muted">Measured ${fmtTime(r.time)} in <code>${esc(r.path)}</code>. "Oldest clip" is how far back recordings go for that camera.
      Snapshots don't count toward the clip limit, but are cleaned up along with the clips. Cleanup runs every 5 minutes.</p>`;
}
$('#cfg-storage-refresh').onclick = () => loadStorage(true);

// ------------------------------------------------------------ account (sign-in)
const errText = e => { const m = String(e.message).match(/^\d+ (.*)$/s); try { return JSON.parse(m[1]).detail; } catch { return e.message; } };
async function loadAccount() {
  const s = await api('/api/auth/state');
  $('#acct-user').textContent = s.username || '';
  $('#acct-hook-state').textContent = s.hook_token ? 'A token is set. Generating a new one replaces it.' : 'No token: external triggers are refused.';
  $('#acct-hook-revoke').classList.toggle('hidden', !s.hook_token);
}
// Collapsed by default; the form is cloned in on open and removed on close (see index.html).
$('#acct-card').ontoggle = () => {
  const slot = $('#acct-pw-slot');
  slot.replaceChildren();
  $('#acct-pw-msg').textContent = '';
  if (!$('#acct-card').open) return;
  slot.append($('#acct-pw-tpl').content.cloneNode(true));
  $('#acct-pw-form').onsubmit = changePassword;
  $('#acct-pw-form [name=current]').focus();
};
async function changePassword(e) {
  e.preventDefault();
  const f = e.target, msg = $('#acct-pw-msg');
  msg.className = 'small'; msg.textContent = '';
  if (f.new.value !== f.repeat.value) { msg.className = 'small err'; msg.textContent = "The new passwords don't match."; return; }
  try {
    await api('/api/auth/password', { method: 'POST', body: { current: f.current.value, new: f.new.value } });
    f.reset(); msg.className = 'small ok'; msg.textContent = 'Password changed. Other devices were signed out.';
  } catch (x) { msg.className = 'small err'; msg.textContent = errText(x); }
}
$('#acct-logout-all').onclick = async () => {
  await api('/api/auth/logout-all', { method: 'POST' });
  $('#acct-pw-msg').className = 'small ok'; $('#acct-pw-msg').textContent = 'All other devices were signed out.';
};
$('#acct-hook-new').onclick = async () => {
  const { token } = await api('/api/auth/hook-token', { method: 'POST' });
  $('#acct-token').textContent = token; $('#acct-token-box').classList.remove('hidden'); loadAccount();
};
$('#acct-hook-revoke').onclick = async () => {
  await api('/api/auth/hook-token', { method: 'DELETE' });
  $('#acct-token-box').classList.add('hidden'); loadAccount();
};
$('#logout').onclick = async () => { await api('/api/auth/logout', { method: 'POST' }).catch(() => {}); location.replace('/login'); };

async function loadLlmModels() {
  const sel = $('#cfg-llm-model'), st = $('#cfg-llm-status');
  st.textContent = 'loading…'; st.className = 'small muted';
  const r = await api(`/api/config/llm-models?url=${encodeURIComponent(S.cfg.llm.url)}`).catch(e => ({ ok: false, error: e.message, models: [] }));
  const cur = S.cfg.llm.model;
  const names = r.models.map(m => m.name);
  if (cur && !names.includes(cur)) names.unshift(cur);
  sel.innerHTML = names.map(n => {
    const m = r.models.find(x => x.name === n);
    return `<option value="${esc(n)}" ${n === cur ? 'selected' : ''}>${esc(n)}${m ? ` (${fmtBytes(m.size)})` : ' (not installed)'}</option>`;
  }).join('');
  if (r.ok) { st.textContent = `✓ Ollama reachable — ${r.models.length} models`; st.className = 'small good'; }
  else { st.textContent = `✗ ${r.error}`; st.className = 'small warn'; }
  const m = r.models.find(x => x.name === cur);
  if (m && m.size > 6e9) { st.textContent += ' · selected model is large; it will compete with detection for GPU memory'; st.className = 'small warn'; }
}
$('#cfg-llm-refresh').onclick = loadLlmModels;

$('#cfg-add-cam').onclick = () => {
  const n = S.cfg.cameras.length + 1;
  S.cfg.cameras.push({ id: `camera_${n}`, name: `Camera ${n}`, brand: 'hikvision', host: '', port: null, username: 'admin', password: '', credential: null,
    channel: 1, main_url: '', sub_url: null, enabled: true, ring_seconds: 300, analyze_fps: 5, detect_fps: 3, always_detect: false,
    record_audio: true });
  S.open = new Set([S.cfg.cameras.length - 1]); renderCams(); markDirty();
  $('#cfg-cams .cam:last-child [data-ck=name]')?.focus();
};
$('#cfg-discard').onclick = () => { S.cfg = JSON.parse(S.saved); S.probe = {}; banner(); renderSetup(); loadLlmModels(); };
$('#cfg-save').onclick = async () => {
  const bad = $$('#tab-setup input:invalid');
  if (bad.length) { bad[0].focus(); banner('err', 'Some fields are invalid — check the highlighted inputs.'); return; }
  const btn = $('#cfg-save'); btn.disabled = true; btn.textContent = 'Saving…';
  try {
    const r = await fetch('/api/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(S.cfg) });
    const j = await r.json();
    if (!r.ok) {
      const errs = j.detail?.errors || [typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail)];
      banner('err', `<b>Not saved:</b><ul>${errs.map(e => `<li>${esc(prettyErr(e))}</li>`).join('')}</ul>`);
      return;
    }
    const probes = S.probe;
    S.dirty = false; await initSetup(true); S.probe = probes; renderCams();
    cameras = await api('/api/cameras'); refreshCameraSelects();
    if (j.restart_required.length) {
      banner('info', `Saved. Restart needed to apply: <b>${esc(j.restart_required.join(', '))}</b>. Camera, search and storage-limit changes are already live.`,
        '<button type="button" class="primary" id="cfg-restart">Restart now</button>');
      $('#cfg-restart').onclick = restartServer;
    } else banner('ok', 'Saved and applied — cameras are (re)connecting.');
  } catch (e) { banner('err', 'Save failed: ' + esc(e.message)); }
  finally { btn.disabled = false; btn.textContent = 'Save configuration'; }
};
function prettyErr(e) {
  return e.replace(/^cameras\.(\d+)\.?/, (_, i) => `Camera “${S.cfg.cameras[+i]?.name || +i + 1}”: `)
    .replace(/^credentials\.(\d+)\.?/, (_, i) => `Saved login “${S.cfg.credentials[+i]?.name || +i + 1}”: `).replace('Value error, ', '');
}
async function restartServer() {
  banner('info', 'Restarting… this page will reconnect automatically.');
  await api('/api/restart', { method: 'POST' }).catch(() => {});
  const port = S.cfg.server.port, proto = S.cfg.server.https ? 'https:' : 'http:';
  const same = proto === location.protocol && String(port) === (location.port || (location.protocol === 'https:' ? '443' : '80'));
  await new Promise(r => setTimeout(r, 2500));
  for (let k = 0; k < 60; k++) {
    try {
      if (!same) { banner('ok', `Restarted. <a href="${proto}//${location.hostname}:${port}/#setup">Open the new address</a>`); return; }
      await api('/api/stats'); location.reload(); return;
    } catch { await new Promise(r => setTimeout(r, 1000)); }
  }
  banner('err', 'Server did not come back — check the terminal/logs.');
}
function refreshCameraSelects() {
  const opts = cameras.map(c => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join('');
  for (const sel of ['#ev-cam', '#tr-cam']) $(sel).innerHTML = '<option value="">All cameras</option>' + opts;
  $('#zn-cam').innerHTML = opts;
}
