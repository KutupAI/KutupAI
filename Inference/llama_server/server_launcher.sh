#!/bin/bash

set -e

LLAMA_SERVER="/workspace/llama.cpp/build/bin/llama-server"
MODEL="/workspace/KutupAI/Inference/models/gemma-4-12b-it-Q4_K_M.gguf"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -c 8192 \
  -ngl 999 \
  -fa on \
  -t 12 \
  --host 0.0.0.0 \
  --port 8080