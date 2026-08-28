#!/usr/bin/env bash
# KutupAI — show status of stack services (ports, PIDs, basic health).
set -u

ROOT="/workspace/KutupAI"
RUN_DIR="${ROOT}/run"

PORT_GEMMA=8082
PORT_PADDLEOCR=8111
PORT_ORCHESTRATION=8000
PORT_APPLICATION=8080
PORT_PRESENTATION=5173

# Mirrors start_all.sh: local Gemma is off unless an agent needs the "local" backend.
ENABLE_LOCAL_GEMMA="${ENABLE_LOCAL_GEMMA:-0}"

port_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]${port}$" >/dev/null 2>&1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:.]${port}$" >/dev/null 2>&1
  else
    (echo >/dev/tcp/127.0.0.1/"${port}") >/dev/null 2>&1
  fi
}

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

listener_pid() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep -E "[:.]${port} " | head -1 \
      | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2
  fi
}

http_code() {
  local url="$1"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' -m 3 "${url}" 2>/dev/null)" || code="000"
  [[ -z "${code}" ]] && code="000"
  echo "${code}"
}

resolve_pid_display() {
  local managed_pid="$1"
  local port="$2"
  local listener
  listener="$(listener_pid "${port}")"
  if pid_alive "${managed_pid}"; then
    echo "${managed_pid} (managed)"
  elif [[ -n "${listener}" ]]; then
    echo "${listener} (port)"
  elif [[ -n "${managed_pid}" ]]; then
    echo "${managed_pid} (stale pidfile)"
  else
    echo "-"
  fi
}

print_row() {
  printf "%-18s %-10s %-22s %-6s %s\n" "$1" "$2" "$3" "$4" "$5"
}

echo "========================================"
echo "KutupAI Status"
echo "========================================"
echo
print_row "SERVICE" "STATE" "PID" "PORT" "HEALTH"
print_row "-------" "-----" "---" "----" "------"

# --- Gemma ---
code="$(http_code "http://127.0.0.1:${PORT_GEMMA}/health")"
pid_d="$(resolve_pid_display "$(read_pidfile "${RUN_DIR}/inference_gemma.pid")" "${PORT_GEMMA}")"
if [[ "${ENABLE_LOCAL_GEMMA}" != "1" ]] && ! port_listening "${PORT_GEMMA}"; then
  print_row "Gemma" "DISABLED" "-" "${PORT_GEMMA}" "n/a (agents use EVREN)"
elif port_listening "${PORT_GEMMA}" && [[ "${code}" =~ ^[123] ]]; then
  print_row "Gemma" "RUNNING" "${pid_d}" "${PORT_GEMMA}" "UP (HTTP ${code})"
elif port_listening "${PORT_GEMMA}"; then
  print_row "Gemma" "LISTEN" "${pid_d}" "${PORT_GEMMA}" "DOWN (HTTP ${code})"
else
  print_row "Gemma" "STOPPED" "${pid_d}" "${PORT_GEMMA}" "DOWN (HTTP ${code})"
fi

# --- PaddleOCR-VL ---
code="$(http_code "http://127.0.0.1:${PORT_PADDLEOCR}/health")"
pid_d="$(resolve_pid_display "$(read_pidfile "${RUN_DIR}/inference_paddleocr.pid")" "${PORT_PADDLEOCR}")"
if port_listening "${PORT_PADDLEOCR}" && [[ "${code}" =~ ^[123] ]]; then
  print_row "PaddleOCR-VL" "RUNNING" "${pid_d}" "${PORT_PADDLEOCR}" "UP (HTTP ${code})"
elif port_listening "${PORT_PADDLEOCR}"; then
  print_row "PaddleOCR-VL" "LISTEN" "${pid_d}" "${PORT_PADDLEOCR}" "DOWN (HTTP ${code})"
else
  print_row "PaddleOCR-VL" "STOPPED" "${pid_d}" "${PORT_PADDLEOCR}" "DOWN (HTTP ${code})"
fi

# --- Orchestration ---
code="$(http_code "http://127.0.0.1:${PORT_ORCHESTRATION}/health")"
pid_d="$(resolve_pid_display "$(read_pidfile "${RUN_DIR}/orchestration.pid")" "${PORT_ORCHESTRATION}")"
if port_listening "${PORT_ORCHESTRATION}" && [[ "${code}" =~ ^[123] ]]; then
  print_row "Orchestration" "RUNNING" "${pid_d}" "${PORT_ORCHESTRATION}" "UP (HTTP ${code})"
elif port_listening "${PORT_ORCHESTRATION}"; then
  print_row "Orchestration" "LISTEN" "${pid_d}" "${PORT_ORCHESTRATION}" "DOWN (HTTP ${code})"
else
  print_row "Orchestration" "STOPPED" "${pid_d}" "${PORT_ORCHESTRATION}" "DOWN (HTTP ${code})"
fi

# --- Application (404 on / is healthy for Drogon) ---
code="$(http_code "http://127.0.0.1:${PORT_APPLICATION}/")"
pid_d="$(resolve_pid_display "$(read_pidfile "${RUN_DIR}/application.pid")" "${PORT_APPLICATION}")"
if port_listening "${PORT_APPLICATION}" && [[ "${code}" != "000" ]]; then
  print_row "Application" "RUNNING" "${pid_d}" "${PORT_APPLICATION}" "UP (HTTP ${code})"
elif port_listening "${PORT_APPLICATION}"; then
  print_row "Application" "LISTEN" "${pid_d}" "${PORT_APPLICATION}" "DOWN (HTTP ${code})"
else
  print_row "Application" "STOPPED" "${pid_d}" "${PORT_APPLICATION}" "DOWN (HTTP ${code})"
fi

# --- Presentation (IPv4 or IPv6) ---
code4="$(http_code "http://127.0.0.1:${PORT_PRESENTATION}/")"
code6="$(http_code "http://[::1]:${PORT_PRESENTATION}/")"
pid_d="$(resolve_pid_display "$(read_pidfile "${RUN_DIR}/presentation.pid")" "${PORT_PRESENTATION}")"
if [[ "${code4}" =~ ^[123] ]]; then
  print_row "Presentation" "RUNNING" "${pid_d}" "${PORT_PRESENTATION}" "UP (HTTP ${code4})"
elif [[ "${code6}" =~ ^[123] ]]; then
  print_row "Presentation" "RUNNING" "${pid_d}" "${PORT_PRESENTATION}" \
    "UP (HTTP ${code6}, [::1] only)"
elif port_listening "${PORT_PRESENTATION}"; then
  print_row "Presentation" "LISTEN" "${pid_d}" "${PORT_PRESENTATION}" "DOWN"
else
  print_row "Presentation" "STOPPED" "${pid_d}" "${PORT_PRESENTATION}" "DOWN"
fi

echo
echo "URLs:"
echo "  Presentation:  http://0.0.0.0:${PORT_PRESENTATION}/"
echo "  Application:   http://127.0.0.1:${PORT_APPLICATION}/"
echo "  Orchestration: http://127.0.0.1:${PORT_ORCHESTRATION}/health"
echo "  Gemma:         http://127.0.0.1:${PORT_GEMMA}/health"
echo "  PaddleOCR-VL:  http://127.0.0.1:${PORT_PADDLEOCR}/health"
echo
echo "PID dir: ${RUN_DIR}/"
echo "Log dir: ${ROOT}/logs/"
