"""Run WriterAgent against the live Inference server.

From the repository root:
    python Tests/Agents/manual_writer_live.py
    python Tests/Agents/manual_writer_live.py --input path\\to\\writer_state.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Agents.writer_agent.agent import WriterAgent


# All personal and institution data below are fictional test values.
DEFAULT_STATE: dict[str, Any] = {
    "request": {
        "success": True,
        "question": "Bu ne sözleşmesi ve hangi kurum ilgilenir?",
        "document": {
            "document_id": "DOC-TR-2026-001",
            "file_name": "Elektrik_Abonelik_Sozlesmesi.pdf",
            "file_type": "pdf",
        },
    },
    "ocr": {
        "success": True,
        "ocr_data": {
            "page_count": 1,
            "language": "tr",
            "pages": [],
            "full_text": (
                "ELEKTRİK ABONELİK SÖZLEŞMESİ\n"
                "Sözleşme No: EAS-2026-001\n"
                "Abone: Ayşe Yılmaz (örnek kayıt)\n"
                "Kullanım yeri: Çankaya, Ankara\n"
                "Sözleşmenin konusu, mesken elektrik aboneliğinin başlatılması; "
                "tüketim, faturalandırma ve tarafların yükümlülüklerinin belirlenmesidir.\n"
                "Tarih: 15.08.2026"
            ),
            "vision": {
                "signature": {"detected": True, "handwritten": True},
                "stamp": {"detected": False},
            },
        },
    },
    "classification": {
        "success": True,
        "document_type": "Elektrik abonelik sözleşmesi",
        "classification_confidence": 0.95,
    },
    "extraction": {
        "success": True,
        "contract_number": "EAS-2026-001",
        "applicant_name": "Ayşe Yılmaz (örnek kayıt)",
        "application_date": "2026-08-15",
        "service_address": "Çankaya, Ankara",
        "service_type": "Mesken elektrik aboneliği",
        "sender": None,
        "date": "2026-08-15",
        "address": "Çankaya, Ankara",
        "phone": None,
        "email": None,
    },
    "validation": {
        "success": True,
        "is_complete": False,
        "errors": [],
        "warnings": ["İletişim telefonu ve e-posta alanları belgede bulunamadı."],
    },
    "rag": {
        "success": True,
        "rag_data": {
            "operation": "retrieve",
            "query": "elektrik abonelik sözleşmesi tüketici hizmetleri",
            "results": [],
        },
    },
    "summary": {
        "success": True,
        "rag_summary_text": (
            "Belge, mesken elektrik aboneliğinin başlatılmasına ilişkin bir elektrik abonelik "
            "sözleşmesidir. Sözleşme; kullanım yeri, tüketim, faturalandırma ve tarafların "
            "yükümlülüklerini düzenler. Belgedeki iletişim bilgileri eksiktir."
        ),
    },
    "routing": {
        "success": True,
        "department": "Enerji Hizmetleri Birimi (örnek yönlendirme)",
    },
    "writing": {},
}
def load_state(path: str | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_STATE
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object matching the Writer state contract.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to a Writer Unified State JSON file.")
    args = parser.parse_args()

    result = WriterAgent().run(load_state(args.input))
    print(json.dumps(result["writing"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
