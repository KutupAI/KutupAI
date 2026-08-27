#!/usr/bin/env bash
# KutupAI — start the full local stack with one command.
# Reuses Inference/*.sh launchers; does not kill unrelated processes.
set -u

ROOT="/workspace/KutupAI"
RUN_DIR="${ROOT}/run"
LOG_DIR="${ROOT}/logs"
VENV="${ROOT}/.venv"
APP_BIN="${ROOT}/Application/build/SmartGovernmentAI_Application"

# --- Ports (match live stack + LlamaClient / Vite proxy) ---
PORT_GEMMA=8082
PORT_PADDLEOCR=8111
PORT_ORCHESTRATION=8000
PORT_APPLICATION=8080
PORT_PRESENTATION=5173

mkdir -p "${RUN_DIR}" "${LOG_DIR}" \
  "${ROOT}/Storage/files/temp_processing"

# shellcheck disable=SC1091
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
else
  echo "[WARN] .venv not found at ${VENV} — Orchestration may fail without it"
fi

export PATH="${VENV}/bin:${PATH}"
export VIRTUAL_ENV="${VENV}"

# OCR on GPU (Paddle 3.x). Skip model-host connectivity check on startup.
export OCR_DEVICE="${OCR_DEVICE:-gpu:0}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export FLAGS_use_mkldnn="${FLAGS_use_mkldnn:-0}"
export PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT="${PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT:-0}"

# ---------------------------------------------------------------------------
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

# True if an existing PID file points at a live process.
managed_running() {
  local name="$1"
  local pid
  pid="$(read_pidfile "${RUN_DIR}/${name}.pid")"
  pid_alive "${pid}"
}

http_code() {
  local url="$1"
  local code
  code="$(curl -sS -o /dev/null -w '%{http_code}' -m 3 "${url}" 2>/dev/null)" || code="000"
  [[ -z "${code}" ]] && code="000"
  echo "${code}"
}

http_ok() {
  local url="$1"
  local code
  code="$(http_code "${url}")"
  [[ "${code}" =~ ^[123][0-9][0-9]$ ]]
}

# Application answers 404 on /; treat any HTTP response as up.
http_up() {
  local url="$1"
  local code
  code="$(http_code "${url}")"
  [[ "${code}" != "000" ]]
}

wait_for() {
  local label="$1"
  local check_fn="$2"
  local timeout="${3:-180}"
  local i=0
  printf "[WAIT] %s (up to %ss)... " "${label}" "${timeout}"
  while (( i < timeout )); do
    if "${check_fn}"; then
      echo "ready"
      return 0
    fi
    sleep 2
    i=$((i + 2))
  done
  echo "TIMEOUT"
  return 1
}

# Node for Vite — PATH first, then Cursor/VS Code bundled binaries (no npm required).
resolve_node() {
  local candidate
  if command -v node >/dev/null 2>&1; then
    command -v node
    return 0
  fi
  for candidate in \
    /root/.cursor-server/bin/linux-x64/*/node \
    /root/.vscode-server/cli/servers/*/server/node; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

start_bg() {
  local name="$1"
  local logfile="$2"
  shift 2
  # Remaining args: command
  nohup "$@" >>"${logfile}" 2>&1 &
  local pid=$!
  echo "${pid}" >"${RUN_DIR}/${name}.pid"
  echo "[START] ${name} pid=${pid} log=${logfile}"
}

# ---------------------------------------------------------------------------
# 1) Gemma (text LLM) — Inference/llama_server/server_launcher.sh → :8082
# ---------------------------------------------------------------------------
start_inference_gemma() {
  local name="inference_gemma"
  local log="${LOG_DIR}/inference_gemma.log"
  local launcher="${ROOT}/Inference/llama_server/server_launcher.sh"

  if managed_running "${name}"; then
    echo "[OK] Gemma inference already managed (pid $(read_pidfile "${RUN_DIR}/${name}.pid")) on port ${PORT_GEMMA}"
    return 0
  fi
  if port_listening "${PORT_GEMMA}"; then
    echo "[OK] Gemma inference already running on port ${PORT_GEMMA}"
    return 0
  fi
  if [[ ! -x "${launcher}" ]] && [[ -f "${launcher}" ]]; then
    chmod +x "${launcher}" || true
  fi
  if [[ ! -f "${launcher}" ]]; then
    echo "[ERROR] Missing ${launcher}"
    return 1
  fi
  if [[ ! -x /workspace/llama.cpp/build/bin/llama-server ]]; then
    echo "[ERROR] llama-server binary not found at /workspace/llama.cpp/build/bin/llama-server"
    return 1
  fi

  echo "[START] Starting Gemma inference on port ${PORT_GEMMA}"
  : >"${log}"
  start_bg "${name}" "${log}" bash "${launcher}"
}

# ---------------------------------------------------------------------------
# 2) PaddleOCR-VL — Inference/start_paddleocr_vl.sh → :8111
# ---------------------------------------------------------------------------
start_inference_paddleocr() {
  local name="inference_paddleocr"
  local log="${LOG_DIR}/inference_paddleocr.log"
  local launcher="${ROOT}/Inference/start_paddleocr_vl.sh"

  if managed_running "${name}"; then
    echo "[OK] PaddleOCR-VL already managed (pid $(read_pidfile "${RUN_DIR}/${name}.pid")) on port ${PORT_PADDLEOCR}"
    return 0
  fi
  if port_listening "${PORT_PADDLEOCR}"; then
    echo "[OK] PaddleOCR-VL already running on port ${PORT_PADDLEOCR}"
    return 0
  fi
  if [[ ! -x "${launcher}" ]] && [[ -f "${launcher}" ]]; then
    chmod +x "${launcher}" || true
  fi
  if [[ ! -f "${launcher}" ]]; then
    echo "[ERROR] Missing ${launcher}"
    return 1
  fi

  echo "[START] Starting PaddleOCR-VL on port ${PORT_PADDLEOCR}"
  : >"${log}"
  start_bg "${name}" "${log}" bash "${launcher}"
}

# ---------------------------------------------------------------------------
# 3) Orchestration — python -m Orchestration.main → :8000
# ---------------------------------------------------------------------------
start_orchestration() {
  local name="orchestration"
  local log="${LOG_DIR}/orchestration.log"

  if managed_running "${name}"; then
    echo "[OK] Orchestration already managed (pid $(read_pidfile "${RUN_DIR}/${name}.pid")) on port ${PORT_ORCHESTRATION}"
    return 0
  fi
  if port_listening "${PORT_ORCHESTRATION}"; then
    echo "[OK] Orchestration already running on port ${PORT_ORCHESTRATION}"
    return 0
  fi
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[ERROR] Python venv missing: ${VENV}/bin/python"
    return 1
  fi

  echo "[START] Starting Orchestration on port ${PORT_ORCHESTRATION}"
  : >"${log}"
  (
    cd "${ROOT}" || exit 1
    export ORCHESTRATION_HOST="${ORCHESTRATION_HOST:-127.0.0.1}"
    export ORCHESTRATION_PORT="${PORT_ORCHESTRATION}"
    # Align agent clients with live Gemma port (Application owns 8080).
    export INFERENCE_HOST="${INFERENCE_HOST:-127.0.0.1}"
    export INFERENCE_PORT="${INFERENCE_PORT:-${PORT_GEMMA}}"
    export INFERENCE_URL="${INFERENCE_URL:-http://127.0.0.1:${PORT_GEMMA}/v1/chat/completions}"
    nohup "${VENV}/bin/python" -m Orchestration.main >>"${log}" 2>&1 &
    echo $! >"${RUN_DIR}/${name}.pid"
  )
  echo "[START] orchestration pid=$(read_pidfile "${RUN_DIR}/${name}.pid") log=${log}"
}

# ---------------------------------------------------------------------------
# OCR pre-warm — load Paddle GPU models once (avoids multi-minute first request).
# Set KUTUPAI_SKIP_OCR_WARMUP=1 to skip.
# ---------------------------------------------------------------------------
warmup_ocr() {
  if [[ "${KUTUPAI_SKIP_OCR_WARMUP:-}" == "1" ]]; then
    echo "[SKIP] OCR warm-up disabled (KUTUPAI_SKIP_OCR_WARMUP=1)"
    return 0
  fi
  if [[ ! -x "${VENV}/bin/python" ]]; then
    return 0
  fi
  echo "[WARM] Pre-loading OCR engine on ${OCR_DEVICE} (cached models = fast)..."
  ( cd "${ROOT}" && "${VENV}/bin/python" - <<'PY' ) >>"${LOG_DIR}/ocr_warmup.log" 2>&1 || {
    echo "[WARN] OCR warm-up failed — see ${LOG_DIR}/ocr_warmup.log"
    return 0
  }
import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from Agents.ocr_agent.engines.paddle_engine import get_shared_engine
from Agents.ocr_agent.config import OCRConfig
cfg = OCRConfig.from_env()
engine = get_shared_engine(cfg)
engine._ensure_pipeline()
print("OCR warm-up OK:", engine.last_engine_name, "device=", cfg.device)
PY
  echo "[WARM] OCR engine ready ($(tail -1 "${LOG_DIR}/ocr_warmup.log" 2>/dev/null || echo 'see log'))"
}

# ---------------------------------------------------------------------------
# 4) Application — Drogon binary → :8080
# ---------------------------------------------------------------------------
start_application() {
  local name="application"
  local log="${LOG_DIR}/application.log"

  if managed_running "${name}"; then
    echo "[OK] Application already managed (pid $(read_pidfile "${RUN_DIR}/${name}.pid")) on port ${PORT_APPLICATION}"
    return 0
  fi
  if port_listening "${PORT_APPLICATION}"; then
    echo "[OK] Application already running on port ${PORT_APPLICATION}"
    return 0
  fi
  if [[ ! -x "${APP_BIN}" ]]; then
    echo "[ERROR] Application binary missing: ${APP_BIN}"
    echo "        Build with: cmake --build ${ROOT}/Application/build --config Release"
    return 1
  fi

  echo "[START] Starting Application on port ${PORT_APPLICATION}"
  : >"${log}"
  (
    cd "${ROOT}/Application" || exit 1
    export ORCHESTRATION_BASE_URL="${ORCHESTRATION_BASE_URL:-http://127.0.0.1:${PORT_ORCHESTRATION}}"
    export APP_TEMP_UPLOAD_ROOT_DIR="${APP_TEMP_UPLOAD_ROOT_DIR:-${ROOT}/Storage/files/temp_processing}"
    export APP_SERVER_PORT="${PORT_APPLICATION}"
    # OCR cold-start + full agent chain can exceed the old 300s default.
    export ORCHESTRATION_TIMEOUT_SECONDS="${ORCHESTRATION_TIMEOUT_SECONDS:-900}"
    nohup "${APP_BIN}" >>"${log}" 2>&1 &
    echo $! >"${RUN_DIR}/${name}.pid"
  )
  echo "[START] application pid=$(read_pidfile "${RUN_DIR}/${name}.pid") log=${log}"
}

# ---------------------------------------------------------------------------
# 5) Presentation — Vite → 0.0.0.0:5173 (proxy /api → :8080)
# ---------------------------------------------------------------------------
start_presentation() {
  local name="presentation"
  local log="${LOG_DIR}/presentation.log"

  if managed_running "${name}"; then
    echo "[OK] Presentation already managed (pid $(read_pidfile "${RUN_DIR}/${name}.pid")) on port ${PORT_PRESENTATION}"
    return 0
  fi
  if port_listening "${PORT_PRESENTATION}"; then
    echo "[OK] Presentation already running on port ${PORT_PRESENTATION}"
    return 0
  fi
  if [[ ! -d "${ROOT}/Presentation/node_modules" ]]; then
    echo "[ERROR] Presentation/node_modules missing — run: cd Presentation && npm install"
    return 1
  fi
  local node_bin vite_bin
  node_bin="$(resolve_node)" || true
  if [[ -z "${node_bin}" ]]; then
    echo "[ERROR] node not found (install Node.js or ensure Cursor/VS Code node is present)"
    return 1
  fi
  vite_bin="${ROOT}/Presentation/node_modules/vite/bin/vite.js"
  if [[ ! -f "${vite_bin}" ]]; then
    echo "[ERROR] Vite missing at ${vite_bin} — run: cd Presentation && npm install"
    return 1
  fi

  echo "[START] Starting Presentation on 0.0.0.0:${PORT_PRESENTATION} (node=${node_bin})"
  : >"${log}"
  (
    cd "${ROOT}/Presentation" || exit 1
    # Run Vite directly — avoids needing npm on PATH.
    nohup "${node_bin}" "${vite_bin}" \
      --host 0.0.0.0 --port "${PORT_PRESENTATION}" --strictPort \
      >>"${log}" 2>&1 &
    echo $! >"${RUN_DIR}/${name}.pid"
  )
  echo "[START] presentation pid=$(read_pidfile "${RUN_DIR}/${name}.pid") log=${log}"
}

# ---------------------------------------------------------------------------
# Health helpers for wait loops
# ---------------------------------------------------------------------------
check_gemma() { http_ok "http://127.0.0.1:${PORT_GEMMA}/health"; }
check_paddle() { http_ok "http://127.0.0.1:${PORT_PADDLEOCR}/health"; }
check_orch() { http_ok "http://127.0.0.1:${PORT_ORCHESTRATION}/health"; }
check_app() { http_up "http://127.0.0.1:${PORT_APPLICATION}/"; }
check_pres() {
  http_ok "http://127.0.0.1:${PORT_PRESENTATION}/" \
    || http_ok "http://[::1]:${PORT_PRESENTATION}/"
}

status_label() {
  if "$1"; then echo "RUNNING"; else echo "DOWN"; fi
}

# ============================= MAIN ========================================
echo "========================================"
echo "KutupAI start_all"
echo "Root: ${ROOT}"
echo "========================================"
echo

ISSUES=()

start_inference_gemma || ISSUES+=("Gemma launch failed")
start_inference_paddleocr || ISSUES+=("PaddleOCR-VL launch failed")

# Inference must be healthy before Orchestration (agents call LLM endpoints).
wait_for "Gemma :${PORT_GEMMA}/health" check_gemma 600 || ISSUES+=("Gemma health timeout")
wait_for "PaddleOCR-VL :${PORT_PADDLEOCR}/health" check_paddle 600 || ISSUES+=("PaddleOCR-VL health timeout")

warmup_ocr || true

start_orchestration || ISSUES+=("Orchestration launch failed")
wait_for "Orchestration :${PORT_ORCHESTRATION}/health" check_orch 60 || ISSUES+=("Orchestration health timeout")

start_application || ISSUES+=("Application launch failed")
wait_for "Application :${PORT_APPLICATION}" check_app 30 || ISSUES+=("Application health timeout")

start_presentation || ISSUES+=("Presentation launch failed")
wait_for "Presentation :${PORT_PRESENTATION}" check_pres 45 || ISSUES+=("Presentation health timeout")

echo
echo "========================================"
echo "KutupAI Startup Summary"
echo "======================="
echo
printf "Application:    %s\n" "$(status_label check_app)"
printf "Orchestration:  %s\n" "$(status_label check_orch)"
printf "Presentation:   %s\n" "$(status_label check_pres)"
printf "Inference Gemma:%s (port %s)\n" "$(status_label check_gemma)" "${PORT_GEMMA}"
printf "Inference OCR:  %s (port %s)\n" "$(status_label check_paddle)" "${PORT_PADDLEOCR}"
echo "======================="
echo
echo "Accessible URLs:"
echo "  Presentation:  http://0.0.0.0:${PORT_PRESENTATION}/  (also http://127.0.0.1:${PORT_PRESENTATION}/)"
echo "  Application:   http://127.0.0.1:${PORT_APPLICATION}/  (API via Presentation /api → this)"
echo "  Orchestration: http://127.0.0.1:${PORT_ORCHESTRATION}/health"
echo "  Gemma LLM:     http://127.0.0.1:${PORT_GEMMA}/health   (internal)"
echo "  PaddleOCR-VL:  http://127.0.0.1:${PORT_PADDLEOCR}/health  (internal)"
echo
echo "Logs: ${LOG_DIR}/"
echo "PIDs: ${RUN_DIR}/"

if ((${#ISSUES[@]} > 0)); then
  echo
  echo "[WARN] Issues:"
  for i in "${ISSUES[@]}"; do
    echo "  - ${i}"
  done
  exit 1
fi

exit 0
