# Agent tests

## OCR Agent
```bash
# from repo root
pip install -r Agents/ocr_agent/requirements.txt
python Tests/Agents/test_ocr_agent.py
# or
pytest Tests/Agents/test_ocr_agent.py -q
```

Tests mock the Paddle engine — they do **not** download OCR models.

## Summary Agent
```bash
# from repo root — live llama-server (Gemma)
python Tests/Agents/manual_test.py

# mocked Integration with Orchestration
pytest Orchestration/tests/test_summary_integration.py -q
```

## Routing Agent
```bash
# from repo root — offline demo (prints full envelope input/output)
python Tests/Agents/manual_test_routing.py

# unit + envelope contract
pytest Tests/Agents/test_routing_agent.py Tests/Agents/test_envelope_contract.py -q

# real RoutingAgent inside Orchestration workflow
pytest Orchestration/tests/test_routing_integration.py -q
```

## Classification Agent
```bash
# from repo root — offline demo (prints full envelope input/output)
python Tests/Agents/manual_test_classification.py

# live Inference Gemma (Inference/llama_server on :8080, gemma3)
python Tests/Agents/manual_test_classification.py --live

# unit + envelope contract (mocked)
pytest Tests/Agents/test_classification_agent.py -q
```

## Extraction Agent
```bash
# from repo root — offline demo (prints full envelope input/output)
python Tests/Agents/manual_test_extraction.py

# live Inference Gemma (Inference/llama_server on :8080, gemma3)
python Tests/Agents/manual_test_extraction.py --live

# unit + envelope contract (mocked LLM)
pytest Tests/Agents/test_extraction_agent.py -q
```
## RAG 

```powershell
# Offline (mock retrieve) — sözleşme şeklini gör
python Tests/RAG/manual_test_rag.py

# Canlı indeks üzerinde gerçek retrieval
python Tests/RAG/manual_test_rag.py --live
```