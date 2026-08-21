@echo off

cd /d "%~dp0.."

echo Starting Llama Server with YaRN...
echo.

llama_server\llama-server.exe ^
    -m "models\gemma3.gguf" ^
    --port 8080 ^
    --ctx-size 131072 ^
    --rope-scaling yarn

pause