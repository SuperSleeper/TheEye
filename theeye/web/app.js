// TheEye NVR UI — vanilla JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts,
    body: opts.body && typeof opts.body !== 'string' ? JSON.stringify(opts.body) : opts.body });
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
  ({ live: initLive, events: loadEvents, triggers: loadTriggers, zones: initZones, rules: initRules, setup: () => initSetup(), search: () => $('#search-q').focus() })[name]?.();
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
      <div style="position:relative"><img alt="" draggable="false"><canvas></canvas><div class="tile-msg">Connecting…</div></div>
      <div class="bar"><span class="grip" title="Drag to reorder">⠿</span><span class="name">${esc(c.name)}</span>
        <span class="badge st"></span><span class="badge recb hidden rec">REC</span>
        <span class="muted ring"></span>
        <button class="recnow">● Record 30s</button>
        <button class="cfg" title="Edit this camera in Setup">⚙</button></div>
    </div>`).join('') : '<p class="muted">No cameras configured — add them on the Setup tab.</p>';
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
      $('.ring', tile).textContent = `loop ${c.ring.seconds}s · ${(c.ring.bytes / 1e6).toFixed(0)} MB`;
      api(`/api/cameras/${encodeURIComponent(id)}/state`).then(s => {
        $('.recb', tile).classList.toggle('hidden', !s.recording);
        drawOverlay($('canvas', tile), $('img', tile), s, id);
      }).catch(() => {});
    }
  };
  timers.live = setInterval(tick, 500); tick();
}

// Live view polls single JPEGs instead of MJPEG: an endless MJPEG response holds one of the browser's
// 6 connections per host, so with 6+ cameras every other request on the page would queue forever.
function pumpFrames(tile) {
  const img = $('img', tile), url = `/api/cameras/${encodeURIComponent(tile.dataset.cam)}/snapshot.jpg`;
  const next = delay => setTimeout(() => {
    if (!tile.isConnected) return;
    if (document.hidden || F.cam) return next(1000);      // paused while a camera is full-window
    img.src = `${url}?t=${Date.now()}`;
  }, delay);
  img.onload = () => { tile.dataset.frames = '1'; next(150); };
  img.onerror = () => { tile.dataset.frames = '0'; next(2000); };
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
  v.muted = false; updateFocusMute();
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
  EV.list = evs.map(e => e.id);                          // newest first, as shown; arrows walk this list
  $$('#ev-list .card').forEach(c => c.onclick = () => openEvent(+c.dataset.id));
}
$('#ev-refresh').onclick = loadEvents; $('#ev-cam').onchange = loadEvents;

// Previous/next in the Events list. The list is newest-first, so "earlier" is the next index.
const EV = { list: [], cur: null };
function eventNav() {
  const i = EV.list.indexOf(EV.cur);
  return { i, earlier: i >= 0 ? EV.list[i + 1] : undefined, later: i > 0 ? EV.list[i - 1] : undefined };
}
function updateEventNav() {
  const n = eventNav();
  $('#pl-prev').disabled = !n.earlier; $('#pl-next').disabled = !n.later;
  $('#pl-pos').textContent = n.i >= 0 ? `${EV.list.length - n.i} / ${EV.list.length}` : '';
  $('#pl-prev').hidden = $('#pl-next').hidden = n.i < 0;   // opened from Search: not part of the list
}
$('#pl-prev').onclick = () => { const n = eventNav(); if (n.earlier) openEvent(n.earlier); };
$('#pl-next').onclick = () => { const n = eventNav(); if (n.later) openEvent(n.later); };
$('#player').addEventListener('keydown', e => {
  if (e.target.matches('input, textarea, select') || e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
    e.preventDefault();                                   // instead of the video's own 5 s seek
    $(e.key === 'ArrowLeft' ? '#pl-prev' : '#pl-next').click();
  }
});
async function openEvent(id, seek) {
  EV.cur = id; updateEventNav();
  const e = await api(`/api/events/${id}`);
  if (EV.cur !== id) return;                              // stepped on to another event meanwhile
  $('#pl-title').textContent = `${camName(e.camera_id)} — ${fmtTime(e.start_ts)} — ${e.rule_name || e.source}`;
  const v = $('#pl-video');
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
      ${esc(d.label)} ${(d.confidence * 100).toFixed(0)}%${d.plate ? ` <span class="chip plate">${esc(d.plate)}</span>` : ''}</span>`).join('') || '<span class="muted">none</span>'}</div>
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

// ---------------------------------------------------------------- search
// Every kind of object found in a recording, with its best certainty; the one that matched is highlighted.
const tagChips = (tags, hit) => (tags || []).map(t =>
  `<span class="chip${t.label === hit ? ' hit' : ''}">${esc(t.label)} ${Math.round(t.conf * 100)}%</span>`).join('');
$('#search-form').onsubmit = async ev => {
  ev.preventDefault();
  const q = $('#search-q').value.trim(); if (!q) return;
  $('#search-meta').textContent = 'Thinking…'; $('#search-results').innerHTML = '';
  try {
    const r = await api('/api/search', { method: 'POST', body: { q } });
    const f = r.filter;
    const parts = [];
    if (f.cameras?.length) parts.push(`cameras=${f.cameras.join(',')}`);
    if (f.labels?.length) parts.push(`labels=${f.labels.join(',')}`);
    if (f.plate) parts.push(`plate=${f.plate}`);
    if (f.zones?.length) parts.push(`zones=${f.zones.join(',')}`);
    if (f.start) parts.push(`from ${f.start}`); if (f.end) parts.push(`to ${f.end}`);
    if (f.hours?.length) parts.push(`hours=${f.hours.join(',')}`);
    if (f.min_confidence) parts.push(`certainty ≥ ${Math.round(f.min_confidence * 100)}%`);
    if (f.visual) parts.push(`looks like “${f.visual}”`);
    $('#search-meta').innerHTML = `${esc(f.explanation || '')} — <code>${esc(r.target)}: ${esc(parts.join(' · ') || 'everything')}</code> · ${r.results.length} results · ${r.ms} ms`;
    $('#search-results').innerHTML = r.results.map(x => {
      if (x.kind === 'detection') return `
        <div class="card" data-ev="${x.event_id || ''}" data-seek="${x.seek ?? ''}">
          ${x.snapshot_path ? `<img loading="lazy" src="/media/${esc(x.snapshot_path)}">` : '<img>'}
          <div class="body"><div class="title"><span>${esc(x.label)} ${(x.confidence * 100).toFixed(0)}%</span><span class="muted">${x.score != null ? 'look-alike ' + x.score.toFixed(2) : ''}</span></div>
          <div class="muted">${esc(camName(x.camera_id))} · ${fmtTime(x.first_ts)}${x.event_id ? '' : ' · no clip'}</div>
          <div class="chips">${tagChips(x.tags, x.label)}${x.plate ? `<span class="chip plate">${esc(x.plate)}</span>` : ''}</div></div></div>`;
      if (x.kind === 'event') return `
        <div class="card" data-ev="${x.id}">${x.snapshot_path ? `<img loading="lazy" src="/media/${esc(x.snapshot_path)}">` : '<img>'}
          <div class="body"><div class="title"><span>Event #${x.id}</span></div><div class="muted">${esc(camName(x.camera_id))} · ${fmtTime(x.start_ts)}</div>
          <div class="chips">${x.tags?.length ? tagChips(x.tags) : x.labels.map(l => `<span class="chip">${esc(l)}</span>`).join('')}${x.plates.map(p => `<span class="chip plate">${esc(p)}</span>`).join('')}</div></div></div>`;
      return `<div class="card"><div class="body"><div class="title">Motion · ${esc(camName(x.camera_id))}</div>
        <div class="muted">${fmtTime(x.start_ts)} · zone ${x.zone_id || 'full'} · ${fmtDur(x.end_ts ? x.end_ts - x.start_ts : null)}</div></div></div>`;
    }).join('') || '<p class="muted">Nothing found.</p>';
    $$('#search-results .card[data-ev]').forEach(c => c.onclick = () => c.dataset.ev && openEvent(+c.dataset.ev, c.dataset.seek ? +c.dataset.seek : undefined));
  } catch (e) { $('#search-meta').textContent = 'Search failed: ' + e.message; }
};

// ---------------------------------------------------------------- triggers
async function loadTriggers() {
  const cam = $('#tr-cam').value;
  const rows = await api(`/api/triggers?limit=500${cam ? '&camera_id=' + encodeURIComponent(cam) : ''}`);
  $('#tr-body').innerHTML = rows.map(r => `<tr><td>${esc(camName(r.camera_id))}</td><td>${esc(r.zone_name || (r.zone_id ? '#' + r.zone_id : 'Full frame'))}</td>
    <td>${fmtTime(r.start_ts)}</td><td>${fmtDur(r.end_ts ? r.end_ts - r.start_ts : null)}</td><td>${(r.peak_area * 100).toFixed(1)}%</td></tr>`).join('');
}
$('#tr-refresh').onclick = loadTriggers; $('#tr-cam').onchange = loadTriggers;

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
    return `<div class="cond" data-i="${i}"><b>${c.type}</b>${body}<select data-k="zone_id">${zoneOptions(c.zone_id)}</select>
      <button type="button" class="danger" data-del>✕</button></div>`;
  }).join('') || '<p class="muted small">Add at least one condition.</p>';
  $$('#rl-conds .cond').forEach(el => {
    const c = R.sel.conditions[+el.dataset.i];
    el.oninput = el.onchange = () => {
      if (c.type === 'object') { c.labels = $$('input[data-k=labels]:checked', el).map(x => x.value); c.min_confidence = +$('[data-k=min_confidence]', el).value; c.min_count = +$('[data-k=min_count]', el).value; }
      if (c.type === 'plate') c.pattern = $('[data-k=pattern]', el).value.toUpperCase();
      const z = $('[data-k=zone_id]', el).value; c.zone_id = z ? +z : null;
    };
    $('[data-del]', el).onclick = () => { R.sel.conditions.splice(+el.dataset.i, 1); renderConds(); };
  });
}
$$('[data-add]').forEach(b => b.onclick = () => {
  const t = b.dataset.add;
  R.sel.conditions.push(t === 'object' ? { type: t, labels: ['person'], zone_id: null, min_confidence: 0.5, min_count: 1 }
    : t === 'plate' ? { type: t, pattern: '*', zone_id: null } : { type: t, zone_id: null });
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
  renderSetup(); loadLlmModels();
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
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v ?? '';
    el.oninput = el.onchange = () => {
      setK(S.cfg, el.dataset.k, el.type === 'checkbox' ? el.checked : el.hasAttribute('data-num') ? +el.value : el.value);
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
  const st = p ? (p.ok ? `<span class="good">✓ ${p.width}×${p.height} ${p.codec}</span>` : '<span class="warn">✗ test failed</span>') : '';
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
    <div class="row"><button type="button" data-dup title="Copy this camera's settings into a new camera">⧉ Duplicate</button>
      <button type="button" class="danger" data-remove>Remove camera</button></div>
  </div>`;
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
  const port = S.cfg.server.port, samePort = String(port) === (location.port || (location.protocol === 'https:' ? '443' : '80'));
  await new Promise(r => setTimeout(r, 2500));
  for (let k = 0; k < 60; k++) {
    try {
      if (!samePort) { banner('ok', `Restarted on port ${port}. <a href="${location.protocol}//${location.hostname}:${port}/#setup">Open the new address</a>`); return; }
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
