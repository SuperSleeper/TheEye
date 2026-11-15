#!/usr/bin/env bash
# Health check for the theeye systemd user service, run every minute by theeye-watchdog.timer.
#
# Crashes are already handled by systemd (Restart=always). This catches the other case: the process
# is alive but the web server has stopped answering. After FAILS_MAX failed checks in a row it
# restarts the service. A service you stopped yourself is left alone.
#
# Usage: watchdog.sh [install-folder]     (default: $THEEYE_HOME, else ~/TheEye-install)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL="${1:-${THEEYE_HOME:-$HOME/TheEye-install}}"
UNIT=theeye.service
FAILS_MAX="${THEEYE_WATCHDOG_FAILS:-3}"
GRACE_SECS="${THEEYE_WATCHDOG_GRACE:-120}"          # give a fresh start time to load models and cameras
STATE="${XDG_RUNTIME_DIR:-/tmp}/theeye-watchdog.fails"

say() { echo "$*"; echo "$(date '+%Y-%m-%d %H:%M:%S,000') WARNING theeye.watchdog: $*" >> "$INSTALL/theeye.log"; }

if ! systemctl --user -q is-active "$UNIT"; then echo 0 > "$STATE"; exit 0; fi

started_us=$(systemctl --user show -p ActiveEnterTimestampMonotonic --value "$UNIT")
uptime_us=$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)
if (( (uptime_us - started_us) / 1000000 < GRACE_SECS )); then echo 0 > "$STATE"; exit 0; fi

# Read the port / https setting from the install's config each time, so config changes are followed.
URL=$("$REPO/.venv/bin/python" - "$INSTALL/config.yaml" <<'PY' 2>/dev/null
import sys, yaml
s = (yaml.safe_load(open(sys.argv[1])) or {}).get("server") or {}
print(f"{'https' if s.get('https', True) else 'http'}://localhost:{s.get('port', 8080)}")
PY
) || URL="https://localhost:8080"

if curl -skf -m 10 -o /dev/null "$URL/api/auth/state"; then echo 0 > "$STATE"; exit 0; fi

fails=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$STATE"
if (( fails < FAILS_MAX )); then
  say "web UI at $URL not answering ($fails/$FAILS_MAX)"
  exit 0
fi
say "web UI at $URL not answering after $fails checks; restarting $UNIT"
echo 0 > "$STATE"
systemctl --user restart "$UNIT"
