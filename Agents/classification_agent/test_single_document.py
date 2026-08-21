"""
test_all_samples.py
----------------------
Runs ClassificationAgent over every *.json file in ocr_samples/ (real
ocr_agent output samples) and prints a summary table at the end.

USAGE:
    python -m Agents.classification_agent.test_all_samples

Requires: llama-server running on port 8092.
"""

from __future__ import annotations

import json
from pathlib import Path

from Agents.classification_agent.agent import ClassificationAgent
from Agents.classification_agent.config import ClassificationConfig

SAMPLES_DIR = Path("Agents/ocr_samples")


def main() -> None:
    files = sorted(SAMPLES_DIR.glob("*.json"))
    if not files:
        print(f"No .json files found in {SAMPLES_DIR}")
        return

    agent = ClassificationAgent(config=ClassificationConfig.from_env())
    rows = []

    for path in files:
        with open(path, encoding="utf-8") as f:
            ocr_response = json.load(f)
        ocr_data = ocr_response["data"]

        state = {
            "document_id": ocr_data.get("document_id", path.stem),
            "ocr_result": ocr_data,
        }

        print(f"--- {path.name} ---")
        result = agent.run(state)["classification_result"]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()

        rows.append(
            {
                "file": path.name,
                "predicted": result.get("document_type"),
                "confidence": result.get("confidence"),
                "status": result.get("status"),
                "seconds": round((result.get("processing_ms") or 0) / 1000, 1),
            }
        )

    print("\n=== OZET ===")
    print(f"{'dosya':<15} {'tahmin':<25} {'guven':<8} {'durum':<15} {'sure'}")
    for r in rows:
        print(f"{r['file']:<15} {str(r['predicted']):<25} {str(r['confidence']):<8} {str(r['status']):<15} {r['seconds']}s")


if __name__ == "__main__":
    main()