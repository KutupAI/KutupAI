"""
test_extreme_weak.py
------------------------
En agir "zayif confidence" testi: 18 taksonomi sinifindan HICBIRINE
gercekten uymayan, resmi bir belge bile olmayan bir metin. Amac:
modelin dogal olarak dusuk confidence uretmesini zorlamak (bkz.
config.min_confidence_threshold=0.50) -- boylece document_type =
diger_belirsiz + en olasi 5 aday davranisini CANLI evren cagrisinda
gozlemlemek.

Proje kokunden calistir: python test_extreme_weak.py
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT / ".env", ROOT / "Agents" / "classification_agent" / ".env"):
    if candidate.is_file():
        load_dotenv(candidate)

from Agents.classification_agent.agent import ClassificationAgent  # noqa: E402

# Kasitli olarak resmi/belge-benzeri HICBIR ozellik tasimayan bir metin --
# ne "Sayin Yetkili" hitabi, ne "talep/sikayet" fiili, ne bir konu basligi.
extreme_text = "asdf kalem masa 123 test test bilmiyorum ne yazacagimi"

agent = ClassificationAgent()
state = {"ocr_result": {"document_id": "EXTREME", "full_text": extreme_text, "pages": []}}
result = agent.run(state)
print(json.dumps(result["classification"], ensure_ascii=False, indent=2))