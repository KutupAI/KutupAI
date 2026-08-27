"""
test_edge_cases.py
---------------------
Gercek evren API'si uzerinden IKI ozel senaryo test eder:

1) YAKIN (ambiguous): iki farkli sinifa da benzer gucte isaret eden,
   kasitli olarak muglak bir metin -- is_ambiguous=true beklenir.
2) ZAYIF (below-threshold): cok kisa/belirsiz, hicbir sinife net
   isaret etmeyen bir metin -- document_type=diger_belirsiz + en
   olasi 5 aday beklenir.

Proje kokunden calistir:
    python test_edge_cases.py

Not: Gercek modelin ne dedigi test'ten test'e degisebilir (deterministik
degil) -- kasitli olarak sinir-durumlara yakin metinler kullanildi ama
model %100 garanti bu esikleri tetiklemeyebilir. Amac gercek davranisi
gozlemlemek.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    raise SystemExit("python -m pip install python-dotenv")

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / ".env", ROOT / "Agents" / "classification_agent" / ".env"):
    if candidate.is_file():
        load_dotenv(candidate)
        print(f"[.env yuklendi] {candidate}")

from Agents.classification_agent.agent import ClassificationAgent  # noqa: E402


def run_case(title: str, full_text: str, agent: ClassificationAgent) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
    state = {
        "ocr_result": {
            "document_id": "EDGE-CASE",
            "full_text": full_text,
            "pages": [],
        }
    }
    result = agent.run(state)
    print(json.dumps(result["classification"], ensure_ascii=False, indent=2))
    print()


def main() -> None:
    agent = ClassificationAgent()

    # --- Senaryo 1: YAKIN -- hem "sikayet" hem "dilekce" gibi okunabilir ---
    # Bir vatandas hem sikayet ediyor hem de bir talepte bulunuyor -- iki
    # sinifin da esit derecede gecerli oldugu klasik sinir-durumu.
    ambiguous_text = (
        "Sayin Yetkili,\n"
        "Mahallemizdeki sokak aydinlatmasi bir haftadir calismiyor. "
        "Bu durumdan rahatsizlik duyuyor ve şikayetçi oluyorum. "
        "Ayni zamanda, arizanin bir an once giderilmesini ve "
        "sokak lambalarinin tamir edilmesini talep ediyorum. "
        "Geregini bilgilerinize sunarim."
    )
    run_case("SENARYO 1: YAKIN (sikayet mi, dilekce mi?)", ambiguous_text, agent)

    # --- Senaryo 2: ZAYIF -- hicbir sinife net isaret etmeyen, cok kisa metin ---
    weak_text = "Merhaba, bilgi icin yaziyorum. Tesekkurler."
    run_case("SENARYO 2: ZAYIF (belirsiz, kisa metin)", weak_text, agent)


if __name__ == "__main__":
    main()