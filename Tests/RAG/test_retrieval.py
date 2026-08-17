"""İnteraktif hukukî retrieval testi için uyumluluk çalıştırıcısı.

Proje kökünden eşdeğer iki komuttan biri kullanılabilir:

    python Tests/RAG/test_retrieval.py
    python -m RAG.scripts.query_retrieval
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RAG.scripts.query_retrieval import main


if __name__ == "__main__":
    main()
