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
