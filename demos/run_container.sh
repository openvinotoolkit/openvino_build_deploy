#!/usr/bin/env bash
set -e
# Usage:
#   run_container.sh [docker run options like -e ... -v ...] <image> [cmd args...]
# Example:
#   run_container.sh -e http_proxy="$http_proxy" -e https_proxy="$https_proxy" -e no_proxy="$no_proxy" paint_your_dreams_demo

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

# Add Intel NPU device if present
[ -e /dev/accel/accel0 ] && DEV_FLAGS+=( --device /dev/accel/accel0 )

# Add host 'video' GID numerically (safer than name)
GFLAGS=()
if VIDEO_GID="$(getent group video 2>/dev/null | cut -d: -f3)"; then
  [ -n "$VIDEO_GID" ] && GFLAGS+=( --group-add "$VIDEO_GID" )
fi

exec docker run --rm -it --init --user root \
  --device /dev/dri \
  -p 7860:7860 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7860 \
  -e DISPLAY -e XAUTHORITY \
  -v "$XAUTHORITY":"$XAUTHORITY":ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  "${GFLAGS[@]}" \
  "${DEV_FLAGS[@]}" \
  "$@"
