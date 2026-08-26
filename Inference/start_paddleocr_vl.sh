#!/bin/bash
set -e

LLAMA_SERVER="/workspace/llama.cpp/build/bin/llama-server"
MODEL="/workspace/KutupAI/Inference/models/PaddleOCR-VL-1.6-GGUF.gguf"
MMPROJ="/workspace/KutupAI/Inference/models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  --mmproj "$MMPROJ" \
  -c 131072 \
  -ngl 999 \
  -t 12 \
  --host 127.0.0.1 \
  --port 8111
