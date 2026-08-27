#!/bin/bash

set -e

LLAMA_SERVER="/workspace/llama.cpp/build/bin/llama-server"
MODEL="/workspace/KutupAI/Inference/models/gemma-4-12b-it-Q4_K_M.gguf"

# Port 8082: Application (Drogon) owns 8080; LlamaClient defaults to :8082.
# Bind localhost only — do not expose the LLM outside the host.
# reasoning-budget 0: Gemma-4 thinking otherwise fills reasoning_content and
# leaves message.content empty (agents then skip classification/summary/writing).
exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -c 8192 \
  -ngl 999 \
  -fa on \
  -t 12 \
  --reasoning-budget 0 \
  --host 127.0.0.1 \
  --port 8082