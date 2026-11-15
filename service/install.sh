#!/usr/bin/env bash
# Install TheEye as a systemd *user* service with a watchdog, so it keeps running after you log out
# or close a terminal, comes back if it crashes, and is restarted if it stops answering.
#
# Usage: service/install.sh [install-folder]     (default: $THEEYE_HOME, else ~/TheEye-install)
#        service/install.sh --uninstall
#
# Afterwards:  systemctl --user status theeye        journalctl --user -u theeye-watchdog
#              systemctl --user restart theeye        (stop: systemctl --user stop theeye)
#              logs keep going to <install-folder>/theeye.log
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ "${1:-}" = "--uninstall" ]; then
  systemctl --user disable --now theeye-watchdog.timer theeye.service 2>/dev/null || true
  rm -f "$UNITS"/theeye.service "$UNITS"/theeye-watchdog.service "$UNITS"/theeye-watchdog.timer
  systemctl --user daemon-reload
  echo "Removed the theeye service and watchdog."
  exit 0
fi

INSTALL="${1:-${THEEYE_HOME:-$HOME/TheEye-install}}"
mkdir -p "$INSTALL" "$UNITS"
INSTALL="$(cd "$INSTALL" && pwd)"
chmod +x "$REPO/run.sh" "$REPO/service/watchdog.sh"

cat > "$UNITS/theeye.service" <<EOF
[Unit]
Description=TheEye camera server
StartLimitIntervalSec=0

[Service]
ExecStart=$REPO/run.sh $INSTALL
Restart=always
RestartSec=10
TimeoutStopSec=20
StandardOutput=append:$INSTALL/theeye.log
StandardError=inherit

[Install]
WantedBy=default.target
EOF

cat > "$UNITS/theeye-watchdog.service" <<EOF
[Unit]
Description=TheEye health check (restarts theeye.service if its web UI stops answering)

[Service]
Type=oneshot
ExecStart=$REPO/service/watchdog.sh $INSTALL
EOF

cat > "$UNITS/theeye-watchdog.timer" <<EOF
[Unit]
Description=Run the TheEye health check every minute

[Timer]
OnBootSec=2min
OnUnitActiveSec=1min
AccuracySec=10s

[Install]
WantedBy=timers.target
EOF

# A copy started by hand (./run.sh) would hold the port and make the service fail to start.
if ! systemctl --user -q is-active theeye.service && pgrep -f "python -m theeye -c config.yaml" >/dev/null; then
  echo "Stopping the copy of TheEye that was started outside the service..."
  pkill -f "python -m theeye -c config.yaml" || true
  for _ in $(seq 20); do pgrep -f "python -m theeye -c config.yaml" >/dev/null || break; sleep 1; done
fi

systemctl --user daemon-reload
systemctl --user enable --now theeye.service theeye-watchdog.timer

if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
  loginctl enable-linger "$USER" 2>/dev/null \
    || echo "Note: run 'sudo loginctl enable-linger $USER' so TheEye starts at boot without logging in."
fi
echo "TheEye is running as a service (logs: $INSTALL/theeye.log)."
