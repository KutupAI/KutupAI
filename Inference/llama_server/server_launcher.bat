@echo off
cd /d "%~dp0.."

echo Starting local Qwen server on port 8082...

llama_server\cuda_bin\llama-server.exe ^
    -m "models\qwen2.5-1.5b-instruct-q4_k_m\qwen2.5-1.5b-instruct-q4_k_m.gguf" ^
    --port 8082 ^
    --ctx-size 32768

pause