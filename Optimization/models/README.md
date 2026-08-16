# ONNX models go here (not committed to Git)

Put the fast-classification ONNX file in this folder:

```
Optimization/models/classification_model.onnx
```

## What stays in Git

| File | Purpose |
|------|---------|
| `model_metadata.json` | Model metadata for the Optimization layer |
| `README.md` | This placeholder |

## What you add locally

| File | Notes |
|------|--------|
| `classification_model.onnx` | Binary — ignored by `*.onnx` in root `.gitignore` |

Used by `services/fast_classification_service.py` for cheap pre-LLM classification.
