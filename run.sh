#!/usr/bin/env bash
# Run TheEye from an install folder that keeps everything private OUTSIDE this repo:
#   <install>/config.yaml   cameras, logins, passwords (created from config.example.yaml on first run)
#   <install>/data/         database, clips, snapshots
#   <install>/*.pt          AI model weights (downloaded there on first run)
# The code always comes from this repo, so edits here take effect on the next restart.
#
# Usage: ./run.sh [install-folder] [theeye options, e.g. -v]
#   default folder: $THEEYE_HOME, else ~/TheEye-install
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then INSTALL="$1"; shift; else INSTALL="${THEEYE_HOME:-$HOME/TheEye-install}"; fi
INSTALL="$(mkdir -p "$INSTALL" && cd "$INSTALL" && pwd)"

case "$INSTALL/" in
  "$REPO"/*) echo "✋ The install folder must be outside the repo ($REPO), or git could pick up your passwords." >&2; exit 1 ;;
esac
chmod 700 "$INSTALL"
if [ ! -f "$INSTALL/config.yaml" ]; then
  cp "$REPO/config.example.yaml" "$INSTALL/config.yaml"
  chmod 600 "$INSTALL/config.yaml"
  echo "Created $INSTALL/config.yaml from the example; add your cameras on the Setup tab."
fi

# Let the git hooks recognise the passwords and details in this install's config (stored in .git/config, not committed).
if ! git -C "$REPO" config --get-all theeye.install 2>/dev/null | grep -qxF "$INSTALL"; then
  git -C "$REPO" config --add theeye.install "$INSTALL" 2>/dev/null || true
fi

cd "$INSTALL"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$REPO/.venv/bin/python" -m theeye -c config.yaml "$@"
