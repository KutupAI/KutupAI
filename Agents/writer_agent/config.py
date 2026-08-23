"""
config.py -- Writer Agent configuration.

This file holds ONLY Writer-Agent-specific tuning knobs (generation
parameters, prompt size limits). It does not, and must not, contain
model paths, server hosts/ports, or anything related to running
gemma3.gguf -- that configuration is owned entirely by the Inference
layer (Inference/configuration/, Inference/models/model_registry.json).
"""

# Generation parameters forwarded into Inference.client.InferenceRequest.
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 512

# Hard cap on characters copied from summary.rag_summary_text into the
# compact model context. Keeps the prompt small even if an upstream
# Agent produces an unexpectedly long summary.
MAX_SUMMARY_CHARS = 2000

# Informational only (matches Inference/models/model_registry.json).
# Not used to load or select a model -- Writer Agent never touches
# the model directly.
MODEL_NAME = "gemma3"
