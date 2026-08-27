#!/usr/bin/env bash
# KutupAI — stop only services started by start_all.sh (via run/*.pid).
# Does NOT kill unrelated Python/Node/llama processes.
set -u

ROOT="/workspace/KutupAI"
RUN_DIR="${ROOT}/run"

# name:port — reverse dependency order
SERVICES=(
  "presentation:5173"
  "application:8080"
  "orchestration:8000"
  "inference_paddleocr:8111"
  "inference_gemma:8082"
)

pid_alive() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

read_pidfile() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    tr -d ' \n\r\t' < "${f}"
  fi
}

listener_pids() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${port} " \
      | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
  fi
}

kill_tree() {
  local pid="$1"
  local children
  children="$(pgrep -P "${pid}" 2>/dev/null || true)"
  if [[ -n "${children}" ]]; then
    local c
    for c in ${children}; do
      kill_tree "${c}"
    done
  fi
  if pid_alive "${pid}"; then
    kill -TERM "${pid}" 2>/dev/null || true
  fi
}

wait_dead() {
  local pid="$1"
  local i=0
  while pid_alive "${pid}" && (( i < 30 )); do
    sleep 0.2
    i=$((i + 1))
  done
  if pid_alive "${pid}"; then
    kill -KILL "${pid}" 2>/dev/null || true
    sleep 0.2
  fi
}

# Soft-stop process tree; then clear any leftover listeners on our port
# (e.g. Vite child orphaned after npm exits). Only runs when a pid file existed.
stop_service() {
  local name="$1"
  local port="$2"
  local pf="${RUN_DIR}/${name}.pid"
  local pid
  pid="$(read_pidfile "${pf}")"

  if [[ -z "${pid}" ]]; then
    echo "[SKIP] ${name}: no pid file (${pf})"
    return 2
  fi

  if pid_alive "${pid}"; then
    kill_tree "${pid}"
    wait_dead "${pid}"
    if pid_alive "${pid}"; then
      echo "[ERROR] ${name}: failed to stop pid ${pid}"
      return 1
    fi
    echo "[STOP] ${name} (pid ${pid})"
  else
    echo "[SKIP] ${name}: pid ${pid} not running"
  fi

  # Reap orphaned children still holding the service port (npm→vite case).
  local lp leftover=0
  for lp in $(listener_pids "${port}"); do
    if pid_alive "${lp}"; then
      echo "[STOP] ${name}: reclaiming port ${port} listener pid ${lp}"
      kill_tree "${lp}"
      wait_dead "${lp}"
      leftover=1
    fi
  done

  rm -f "${pf}"
  return 0
}

echo "========================================"
echo "KutupAI stop_all"
echo "========================================"
echo

STOPPED=0
MISSING=0
FAILED=0

for entry in "${SERVICES[@]}"; do
  name="${entry%%:*}"
  port="${entry##*:}"
  set +e
  stop_service "${name}" "${port}"
  rc=$?
  set -e
  if (( rc == 0 )); then
    STOPPED=$((STOPPED + 1))
  elif (( rc == 2 )); then
    MISSING=$((MISSING + 1))
  else
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "Stopped (had pid files): ${STOPPED}"
echo "No pid file / already gone: ${MISSING}"
if (( FAILED > 0 )); then
  echo "Failed: ${FAILED}"
  exit 1
fi
echo "Done. Unrelated processes were not touched."
exit 0
