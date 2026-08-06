# llama-server binaries go here (not committed to Git)

Place the official `llama.cpp` / `llama-server` Windows (or Linux) build outputs in this folder.

## Required files (typical Windows CPU build)

- `llama-server.exe`
- `llama.dll`, `ggml.dll`, `ggml-base.dll`, and the matching `ggml-cpu-*.dll` set
- related `llama-*-impl.dll` / `mtmd.dll` / OpenMP DLL if your build ships them

## Keep in Git (already in this folder)

- `server_launcher.bat` / `server_launcher.sh` — how to start the server
- `build_config.cmake` — build notes/flags
- this `README.md`

## After copying binaries

1. Put `gemma3.gguf` (or your model) in `../models/` — see `Inference/models/README.md`
2. Run `server_launcher.bat` (Windows) or `server_launcher.sh` (Linux/macOS)
3. Default: `127.0.0.1:8080`

Do **not** commit `.exe` / `.dll` / `output.log` — they are listed in the root `.gitignore`.
