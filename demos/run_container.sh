#!/usr/bin/env bash
set -e

[[ $# -ge 1 ]] || { echo "Usage: $0 <image> [cmd args…]"; exit 2; }
IMG="$1"; shift
CMD=( "$@" )

: "${DISPLAY:?DISPLAY not set}"
XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}
[ -r "$XAUTHORITY" ] || { echo "XAUTHORITY not readable: $XAUTHORITY"; exit 1; }

# Free the camera if something has it open (best-effort)
command -v fuser >/dev/null && fuser -k /dev/video0 /dev/video1 2>/dev/null || true

# Collect available V4L2 devices
DEV_FLAGS=()
for d in /dev/video[0-9]* /dev/media[0-9]*; do
  [ -e "$d" ] && DEV_FLAGS+=( --device "$d" )
done

# Add host 'video' GID numerically (safer than name)
VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3 || true)"
GFLAGS=()
[ -n "$VIDEO_GID" ] && GFLAGS+=( --group-add "$VIDEO_GID" )

exec docker run --rm -it --init --user root \
  --device /dev/dri \
  -p 7860:7860 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7860 \
  -e DISPLAY -e XAUTHORITY \
  -v "$XAUTHORITY":"$XAUTHORITY":ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "${DEV_FLAGS[@]}" \
  "$IMG" "${CMD[@]}"
