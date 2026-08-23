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
