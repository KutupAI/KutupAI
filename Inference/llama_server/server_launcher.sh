#!/bin/bash
cd "$(dirname "$0")"
./llama-server -m ../models/gemma3.gguf -c 8192 -t 12 --host 127.0.0.1 --port 8080