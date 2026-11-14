"""Login for the web UI.

- One account, stored in auth.json next to config.yaml (the install folder, never the repo). The password is kept
  only as a salted scrypt hash; nothing reversible is stored.
- Sessions are random 256-bit tokens in an HttpOnly, Secure, SameSite=Strict cookie. The database stores only a
  SHA-256 of each token, so a copied database can't be used to sign in. Sessions survive restarts.
- Repeated wrong passwords lock that address out with growing delays (and slow everyone a little).
- Requests that change things must come from this site's own pages (Origin check), on top of SameSite.
- External triggers (POST /api/hooks/...) can use a separate bearer token instead of a session.
- First account: created from this machine's browser (localhost) or with `./run.sh --set-password`.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

log = logging.getLogger(__name__)

SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}          # ~32 MB, ~50-100 ms per check
REMEMBER_SECS = 30 * 86400                       # "keep me signed in"
SESSION_SECS = 12 * 3600                         # otherwise: until the browser closes, at most 12 h
MIN_PW, MAX_PW = 8, 256
FAIL_FREE, LOCK_BASE, LOCK_MAX = 5, 30, 3600     # 5 free tries, then 30 s, 60 s, ... up to 1 h
PUBLIC = {"/login", "/login.js", "/style.css", "/favicon.ico",
          "/api/auth/state", "/api/auth/login", "/api/auth/setup"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, maxmem=128 * 2 ** 20, dklen=32, **SCRYPT)
    b = lambda x: base64.b64encode(x).decode()          # noqa: E731
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${b(salt)}${b(h)}"


def check_password(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             maxmem=128 * 2 ** 20, dklen=32)
        return hmac.compare_digest(got, base64.b64decode(h))
    except (ValueError, TypeError):
        return False


_DUMMY = None


def _dummy_hash() -> str:                       # compared against for unknown usernames: same timing as a real one
    global _DUMMY
    _DUMMY = _DUMMY or hash_password(secrets.token_hex(16))
    return _DUMMY


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def password_problem(username: str, pw: str) -> str | None:
    if not username.strip() or len(username) > 64:
        return "Enter a username (up to 64 characters)."
    if len(pw) < MIN_PW:
        return f"Use at least {MIN_PW} characters."
    if len(pw) > MAX_PW:
        return f"Use at most {MAX_PW} characters."
    if pw.lower() == username.strip().lower() or len(set(pw)) < 4:
        return "That password is too easy to guess."
    return None


def is_loopback(host: str | None) -> bool:
    try:
        return ipaddress.ip_address((host or "").split("%")[0]).is_loopback
    except ValueError:
        return False


class AuthStore:
    def __init__(self, path: Path, db):
        self.path, self.db = path, db
        self.lock = threading.Lock()
        self.hash_slots = threading.Semaphore(2)     # caps CPU/RAM spent on password checks
        self.fails: dict[str, list[float]] = {}      # ip -> [failures, locked_until]
        self._data, self._mtime = {}, None
        db.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY, created REAL NOT NULL, expires REAL NOT NULL,
            last_seen REAL NOT NULL, ip TEXT, agent TEXT)""")
        db.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))

    # ------------------------------------------------------------ account file
    def data(self) -> dict:
        """auth.json, re-read when it changes (e.g. after `--set-password` while running)."""
        try:
            m = self.path.stat().st_mtime_ns
        except FileNotFoundError:
            self._data, self._mtime = {}, None
            return {}
        if m != self._mtime:
            try:
                self._data, self._mtime = json.loads(self.path.read_text()), m
            except (ValueError, OSError):
                log.error("%s is unreadable; nobody can sign in until it's fixed", self.path)
                self._data, self._mtime = {"broken": True}, m
        return self._data

    def configured(self) -> bool:
        d = self.data()
        return bool(d.get("password") or d.get("broken"))

    def save(self, **changes) -> None:
        with self.lock:
            d = {**self.data(), **changes}
            d = {k: v for k, v in d.items() if v is not None}
            write_private_json(self.path, d)

    # ------------------------------------------------------------ passwords
    def verify(self, username: str, pw: str) -> bool:
        d = self.data()
        ok_user = hmac.compare_digest(username.strip().lower().encode(), str(d.get("username", "")).lower().encode())
        with self.hash_slots:
            ok_pw = check_password(pw, d.get("password") or _dummy_hash())
        return ok_user and ok_pw and bool(d.get("password"))

    def locked_for(self, ip: str) -> float:
        f = self.fails.get(ip)
        return max(0.0, f[1] - time.time()) if f else 0.0

    def failed(self, ip: str) -> None:
        with self.lock:
            f = self.fails.setdefault(ip, [0, 0.0])
            f[0] += 1
            if f[0] >= FAIL_FREE:
                f[1] = time.time() + min(LOCK_MAX, LOCK_BASE * 2 ** (f[0] - FAIL_FREE))
            if len(self.fails) > 10000:                       # don't let random addresses grow this forever
                self.fails = dict(sorted(self.fails.items(), key=lambda kv: -kv[1][1])[:5000])
        log.warning("failed sign-in from %s (%d in a row)", ip, f[0])

    # ------------------------------------------------------------ sessions
    def new_session(self, ip: str, agent: str, remember: bool) -> str:
        token = secrets.token_urlsafe(32)
        now = time.time()
        self.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                        (_sha(token), now, now + (REMEMBER_SECS if remember else SESSION_SECS), now, ip, agent[:200]))
        return token

    def session(self, token: str | None) -> bool:
        if not token or len(token) > 100 or not self.configured():
            return False
        h, now = _sha(token), time.time()
        row = self.db.one("SELECT expires, last_seen FROM sessions WHERE token_hash=?", (h,))
        if not row or row["expires"] < now:
            return False
        if now - row["last_seen"] > 60:
            self.db.execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (now, h))
        return True

    def end_session(self, token: str | None) -> None:
        if token:
            self.db.execute("DELETE FROM sessions WHERE token_hash=?", (_sha(token),))

    def end_all_sessions(self, keep: str | None = None) -> None:
        self.db.execute("DELETE FROM sessions WHERE token_hash != ?", (_sha(keep) if keep else "",))

    def hook_token_ok(self, auth_header: str | None) -> bool:
        want = self.data().get("hook_token")
        if not want or not auth_header or not auth_header.lower().startswith("bearer "):
            return False
        return hmac.compare_digest(_sha(auth_header[7:].strip()), want)


def write_private_json(path: Path, d: dict) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(d, f, indent=1)
    os.chmod(tmp, 0o600)
    tmp.replace(path)


# ---------------------------------------------------------------- middleware
class AuthMiddleware:
    """Pure ASGI (doesn't buffer the endless live-video responses). Also adds security headers."""

    def __init__(self, app, store: AuthStore):
        self.app, self.store = app, store

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        req = Request(scope)
        path, method = scope["path"], scope["method"]
        https = scope.get("scheme") == "https"

        async def send_with_headers(msg):
            if msg["type"] == "http.response.start":
                h = msg.setdefault("headers", [])
                h += [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                      (b"referrer-policy", b"same-origin"),
                      (b"content-security-policy", b"frame-ancestors 'none'; base-uri 'self'; form-action 'self'")]
                if path.startswith("/api/auth/"):
                    h.append((b"cache-control", b"no-store"))
            await send(msg)

        def deny(resp):
            return resp(scope, receive, send_with_headers)

        if method not in SAFE_METHODS and not same_origin(req):
            return await deny(JSONResponse({"detail": "cross-site request blocked"}, 403))
        if path in PUBLIC or self.store.session(req.cookies.get(cookie_name(https))):
            return await self.app(scope, receive, send_with_headers)
        if path.startswith("/api/hooks/") and self.store.hook_token_ok(req.headers.get("authorization")):
            return await self.app(scope, receive, send_with_headers)
        if path.startswith(("/api/", "/media/")):
            return await deny(JSONResponse({"detail": "sign in required"}, 401))
        return await deny(RedirectResponse("/login", 303))


def cookie_name(https: bool) -> str:
    return "__Host-theeye" if https else "theeye"    # __Host-: browser enforces Secure, Path=/, no Domain


def same_origin(req: Request) -> bool:
    """Browsers always send Origin on cross-site POST/PUT/DELETE. Non-browser clients (curl, Home Assistant)
    send neither header and can't carry our SameSite cookie anyway, so they still need a token."""
    src = req.headers.get("origin") or req.headers.get("referer")
    if not src:
        return True
    host = req.headers.get("host", "")
    return src != "null" and urlsplit(src).netloc.lower() == host.lower()


# ---------------------------------------------------------------- routes
def register(app: FastAPI, store: AuthStore, web: Path) -> None:
    def client_ip(req: Request) -> str:
        return req.client.host if req.client else "?"

    def set_cookie(resp, req: Request, token: str, remember: bool):
        https = req.url.scheme == "https"
        resp.set_cookie(cookie_name(https), token, max_age=REMEMBER_SECS if remember else None, path="/",
                        httponly=True, secure=https, samesite="strict")

    @app.get("/login")
    def login_page():
        return FileResponse(web / "login.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/auth/state")
    def state(req: Request):
        signed_in = store.session(req.cookies.get(cookie_name(req.url.scheme == "https")))
        configured = store.configured()
        return {"configured": configured, "signed_in": signed_in,
                "can_setup": not configured and is_loopback(client_ip(req)),
                "username": store.data().get("username") if signed_in else None,
                "hook_token": bool(store.data().get("hook_token")) if signed_in else None}

    @app.post("/api/auth/setup")
    def setup(req: Request, body: dict = Body(...)):
        if not is_loopback(client_ip(req)):
            raise HTTPException(403, "Create the first account from this computer's browser, "
                                     "or run ./run.sh --set-password")
        user, pw = str(body.get("username", "")), str(body.get("password", ""))
        with store.lock:                                   # first come, first served — only once
            if store.configured():
                raise HTTPException(409, "An account already exists.")
            if p := password_problem(user, pw):
                raise HTTPException(422, p)
            write_private_json(store.path, {"username": user.strip(), "password": hash_password(pw)})
        log.info("account created")
        resp = JSONResponse({"ok": True})
        set_cookie(resp, req, store.new_session(client_ip(req), req.headers.get("user-agent", ""), False), False)
        return resp

    @app.post("/api/auth/login")
    def login(req: Request, body: dict = Body(...)):
        ip = client_ip(req)
        if (wait := store.locked_for(ip)) > 0:
            raise HTTPException(429, f"Too many wrong passwords. Try again in {int(wait) + 1} s.")
        user, pw = str(body.get("username", ""))[:64], str(body.get("password", ""))[:MAX_PW]
        remember = bool(body.get("remember"))
        if not store.verify(user, pw):
            store.failed(ip)
            time.sleep(0.3 + secrets.randbelow(300) / 1000)
            raise HTTPException(401, "Wrong username or password.")
        store.fails.pop(ip, None)
        log.info("signed in from %s", ip)
        resp = JSONResponse({"ok": True})
        set_cookie(resp, req, store.new_session(ip, req.headers.get("user-agent", ""), remember), remember)
        return resp

    @app.post("/api/auth/logout")
    def logout(req: Request):
        https = req.url.scheme == "https"
        store.end_session(req.cookies.get(cookie_name(https)))
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(cookie_name(https), path="/", secure=https, httponly=True, samesite="strict")
        return resp

    @app.post("/api/auth/logout-all")
    def logout_all(req: Request):
        store.end_all_sessions(keep=req.cookies.get(cookie_name(req.url.scheme == "https")))
        return {"ok": True}

    @app.post("/api/auth/password")
    def change_password(req: Request, body: dict = Body(...)):
        ip = client_ip(req)
        if (wait := store.locked_for(ip)) > 0:
            raise HTTPException(429, f"Too many wrong passwords. Try again in {int(wait) + 1} s.")
        d = store.data()
        if not store.verify(d.get("username", ""), str(body.get("current", ""))[:MAX_PW]):
            store.failed(ip)
            raise HTTPException(403, "Your current password is wrong.")
        new = str(body.get("new", ""))
        if p := password_problem(d.get("username", ""), new):
            raise HTTPException(422, p)
        store.save(password=hash_password(new))
        store.end_all_sessions(keep=req.cookies.get(cookie_name(req.url.scheme == "https")))
        log.info("password changed; other sessions signed out")
        return {"ok": True}

    @app.post("/api/auth/hook-token")
    def new_hook_token():
        token = secrets.token_urlsafe(32)
        store.save(hook_token=_sha(token))
        return {"token": token}                # shown once; only its hash is kept

    @app.delete("/api/auth/hook-token")
    def revoke_hook_token():
        store.save(hook_token=None)
        return {"ok": True}

    # Starlette runs middleware outermost-last; added here so it wraps every route and the static files.
    app.add_middleware(AuthMiddleware, store=store)


def set_password_cli(auth_path: Path, db_path: Path | None) -> None:
    """`python -m theeye --set-password`: create the account or reset a forgotten password."""
    import getpass
    import sqlite3
    d = json.loads(auth_path.read_text()) if auth_path.exists() else {}
    user = input(f"Username [{d.get('username', 'admin')}]: ").strip() or d.get("username", "admin")
    while True:
        pw = getpass.getpass("New password: ")
        if p := password_problem(user, pw):
            print(p)
            continue
        if getpass.getpass("Repeat it: ") != pw:
            print("They don't match.")
            continue
        break
    write_private_json(auth_path, {**{k: v for k, v in d.items() if k != "broken"},
                                   "username": user, "password": hash_password(pw)})
    if db_path and db_path.exists():               # a reset signs everyone out
        with sqlite3.connect(db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, created REAL NOT NULL, "
                      "expires REAL NOT NULL, last_seen REAL NOT NULL, ip TEXT, agent TEXT)")
            c.execute("DELETE FROM sessions")
    print(f"Saved to {auth_path}. Sign in at the web UI (a running TheEye picks this up immediately).")
