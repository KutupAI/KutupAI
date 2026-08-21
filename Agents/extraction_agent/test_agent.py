"""
-------------
Bağımsız duman testi - doğrudan çalıştırın (demo için pytest gerekmez):

python -m extraction_agent.test_agent

OCR + Sınıflandırmadan sonra Denetleyicinin yapacağı gibi sahte bir `durum` oluşturur,
ExtractionAgent.run() çalıştırır ve elde edilen JSON'u düzgün bir şekilde yazdırır.
EXTRACTION_LLM_API_KEY ayarlanmamış / dönüştürücüler yüklenmemiş olsa bile çalışır -
regex katmanı alanları yine de dolduracak ve meta.warnings neyin atlandığını açıklayacaktır, bu da entegrasyon testi sırasında tam olarak görmek istediğiniz şeydir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv

    # .env dosyasi bu paketin (extraction_agent/) icinde bulunuyor.
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    print("Uyari: python-dotenv kurulu degil (`pip install python-dotenv`) - "
          ".env dosyasi otomatik yuklenmeyecek, degiskenleri elle export etmeniz gerekir.")

from .agent import ExtractionAgent

logging.basicConfig(level=logging.INFO)

SAMPLE_OCR_TEXT = """\
Sayı: 2024/4521
Tarih: 12.05.2024

Sayın Yetkili,

Elektrik faturam beklediğimden yüksek geldi. İncelenmesini istiyorum.

Ahmet Yılmaz
Tel: 0532 123 45 67
E-posta: ahmet.yilmaz@example.com
Enerji Müdürlüğü'ne başvurulmuştur.
"""


def build_fake_state() -> dict:
    return {
        "ocr_result": {
            "full_text": SAMPLE_OCR_TEXT,
            "lines": SAMPLE_OCR_TEXT.strip().split("\n"),
            "has_signature": False,
            "has_handwritten_signature": False,
        },
        "classification_result": {"document_type": "Şikayet"},
    }


def main() -> None:
    agent = ExtractionAgent()
    state = build_fake_state()
    result_state = agent.run(state)
    print(json.dumps(result_state["extraction_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()