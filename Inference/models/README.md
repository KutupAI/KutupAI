# Model weights go here (not committed to Git)

Put the GGUF file in this folder:

```
Inference/models/gemma3.gguf
```

## What stays in Git

| File | Purpose |
|------|---------|
| `model_registry.json` | Model name, path, quantization, context size |
| `README.md` | This placeholder |

## What you add locally

| File | Notes |
|------|--------|
| `gemma3.gguf` | Binary weights — ignored by `*.gguf` in root `.gitignore` |

Update `model_registry.json` if you rename the file or switch models.
