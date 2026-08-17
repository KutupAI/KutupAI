@echo off
cd /d "%~dp0"
llama-server.exe -m ..\models\gemma3.gguf -c 8192 -t 12 --host 127.0.0.1 --port 8080
pause