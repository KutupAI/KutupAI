@echo off
cd /d "%~dp0"

echo Starting PaddleOCR-VL-1.6...
echo Endpoint: http://127.0.0.1:8111/v1/chat/completions
echo.

llama_server\llama-server.exe ^
    -m "models\PaddleOCR-VL-1.6-GGUF.gguf" ^
    --mmproj "models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf" ^
    --chat-template-file "models\chat_template.jinja" ^
    --port 8111 ^
    --temp 0 ^
    -c 32768 ^
    -np 1 ^
    -ngl 99 ^
    --special

pause