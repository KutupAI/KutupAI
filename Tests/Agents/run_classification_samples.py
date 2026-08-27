"""
run_classification_samples.py
--------------------------------
Agents/ocr_samples/ klasorundeki her ornegi TEK TEK, gercek
ClassificationAgent uzerinden (gercek evren API cagrisiyla) calistirir ve
sonuclarini ayri ayri, net sekilde ayirarak yazdirir.

Proje kokunden calistir:
    python run_classification_samples.py

NOT: config.py artik .env dosyasini otomatik yuklemiyor (onceki
surumde vardi, kaldirilmis). Bu yuzden asagida python-dotenv ile
elle yukluyoruz -- yoksa EVREN_LLM_API_KEY vb. os.environ'a hic
girmez ve evren cagrisi "API key eksik" hatasi verir.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit(
        "python-dotenv kurulu degil. Once calistir: python -m pip install python-dotenv"
    )

PROJECT_ROOT = Path(__file__).resolve().parent

# Olasi .env konumlarinin hepsini dene (proje kokunde mi, yoksa
# Agents/classification_agent/ altinda mi oldugundan emin degiliz) --
# hangisi varsa onu yukler, hicbiri yoksa sessizce devam eder (o zaman
# EVREN_LLM_API_KEY zaten baska bir yerden -- ör. sistem ortam degiskeni
# olarak -- gelmis olmali).
for candidate in (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "Agents" / "classification_agent" / ".env",
    PROJECT_ROOT / "Config" / "secrets.env",
):
    if candidate.is_file():
        load_dotenv(candidate)
        print(f"[.env yuklendi] {candidate}")

from Agents.classification_agent.agent import ClassificationAgent  # noqa: E402

SAMPLES_DIR = PROJECT_ROOT / "Agents" / "ocr_samples"
SAMPLE_FILES = ["fatura.json", "karanlik.json", "imza.json", "normal.json", "pdf.json"]


def run_one(sample_name: str, agent: ClassificationAgent) -> None:
    path = SAMPLES_DIR / sample_name
    if not path.is_file():
        print(f"  [ATLANDI] {path} bulunamadi.")
        return

    with path.open(encoding="utf-8") as f:
        ocr_data = json.load(f)

    result = agent.run({"ocr_result": ocr_data})
    print(json.dumps(result["classification"], ensure_ascii=False, indent=2))


def main() -> None:
    agent = ClassificationAgent()
    for sample_name in SAMPLE_FILES:
        print("=" * 70)
        print(f"ORNEK: {sample_name}")
        print("=" * 70)
        run_one(sample_name, agent)
        print()


if __name__ == "__main__":
    main()