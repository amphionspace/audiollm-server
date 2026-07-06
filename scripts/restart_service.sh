#!/usr/bin/env bash
# Restart the public-facing audiollm-demo systemd service and show the latest
# logs so you can verify the new code is live.
#
# Usage:
#   scripts/restart_service.sh            # restart + tail last 30 log lines
#   scripts/restart_service.sh -f         # restart + follow logs (Ctrl+C to quit)
#   SERVICE=my-demo scripts/restart_service.sh
set -euo pipefail

SERVICE="${SERVICE:-audiollm-demo}"
SERVICE="${SERVICE%.service}"
UNIT="${SERVICE}.service"
FOLLOW=0
for arg in "$@"; do
  case "$arg" in
    -f|--follow) FOLLOW=1 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not available on this host." >&2
  exit 1
fi

unit_file_exists() {
  local dir
  for dir in \
    /etc/systemd/system \
    /run/systemd/system \
    /usr/local/lib/systemd/system \
    /usr/lib/systemd/system \
    /lib/systemd/system; do
    [ -e "${dir}/${UNIT}" ] && return 0
  done
  return 1
}

print_related_units() {
  local dir file printed=0
  shopt -s nullglob
  for dir in \
    /etc/systemd/system \
    /run/systemd/system \
    /usr/local/lib/systemd/system \
    /usr/lib/systemd/system \
    /lib/systemd/system; do
    for file in "${dir}"/*.service; do
      case "${file##*/}" in
        *audio*|*llm*|*asr*|*vllm*|*k2*|*qwen*|*amphion*|*demo*)
          if [ "${printed}" -eq 0 ]; then
            echo "Related installed service units:" >&2
            printed=1
          fi
          echo "  ${file##*/}" >&2
          ;;
      esac
    done
  done
  shopt -u nullglob
}

if ! unit_file_exists; then
  echo "Service '${SERVICE}' is not installed." >&2
  echo "Expected unit file named '${UNIT}' in a systemd unit directory." >&2
  print_related_units
  echo "Set SERVICE=<unit-name> if this deployment uses a different service." >&2
  exit 1
fi

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

echo "==> Restarting ${SERVICE}..."
${SUDO} systemctl restart "${SERVICE}"

sleep 1

STATUS="$(${SUDO} systemctl is-active "${SERVICE}" || true)"
echo "==> Status: ${STATUS}"
if [ "${STATUS}" != "active" ]; then
  echo "Service failed to start. Recent logs:" >&2
  ${SUDO} journalctl -u "${SERVICE}" -n 50 --no-pager >&2 || true
  exit 1
fi

if [ "${FOLLOW}" -eq 1 ]; then
  echo "==> Following logs (Ctrl+C to quit):"
  exec ${SUDO} journalctl -u "${SERVICE}" -f --no-pager
else
  echo "==> Last 30 log lines:"
  ${SUDO} journalctl -u "${SERVICE}" -n 30 --no-pager
fi
