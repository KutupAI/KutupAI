"""Metin chunk'larından kaynak kanıtlı, genişletilebilir hukukî olgu kaydı üretir."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from RAG.configuration.rag_config_loader import amendment_ledger_config, facts_registry_config


_COURT_DECISION = re.compile(
    r"Anayasa\s+Mahkemesi(?:nin|[’']nin)?\s+(\d{1,2}/\d{1,2}/\d{4})\s+tarihli\s+ve\s+"
    r"E\.?\s*[:.]?\s*(\d{4}/\d+)\s*[,;]?\s*K\.?\s*[:.]?\s*(\d{4}/\d+)",
    re.IGNORECASE,
)
_DURATION = re.compile(r"\b(\d{1,3})\s*(saat|gün|ay|yıl)\s+içinde\b", re.IGNORECASE)
_LAW_REFERENCE = re.compile(r"\b(\d{3,5})\s+sayılı\s+Kanun\b", re.IGNORECASE)
_KHK_REFERENCE = re.compile(r"\bKHK\s*[-/]?\s*(\d{2,4})\b", re.IGNORECASE)
_AMENDMENT = re.compile(
    r"(?:Ek|Değişik|Mülga|Yeniden\s+düzenleme|İptal)\s*:\s*"
    r"([^\n]{0,160}?(?:KHK|Kanun)[^\n]{0,120})",
    re.IGNORECASE,
)


def _record_id(fact_type: str, values: dict[str, Any], chunk_id: str) -> str:
    raw = json.dumps([fact_type, values, chunk_id], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _evidence(row: dict[str, Any]) -> dict[str, Any]:
    meta = dict(row.get("metadata") or {})
    return {
        "chunk_id": str(row.get("chunk_id") or meta.get("chunk_id") or "unknown"),
        "source_file": str(meta.get("source_file") or "unknown"),
        "source_type": str(meta.get("source_type") or "unknown"),
        "document_category": str(meta.get("document_category") or "unknown"),
        "law_number": str(meta.get("law_number") or "unknown"),
        "article_no": str(meta.get("article_no") or meta.get("article_number") or "unknown"),
        "page_start": meta.get("page_start") or meta.get("page"),
        "page_end": meta.get("page_end") or meta.get("page"),
    }


def _fact(fact_type: str, values: dict[str, Any], row: dict[str, Any], evidence_text: str) -> dict[str, Any]:
    evidence = _evidence(row)
    return {
        "fact_id": _record_id(fact_type, values, evidence["chunk_id"]),
        "fact_type": fact_type,
        "values": values,
        "evidence": evidence,
        "evidence_text": " ".join(evidence_text.split())[:1200],
        "confidence": 1.0,
        "extraction_method": "deterministic_text_pattern",
    }


def extract_facts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yeni belge türleri ham chunk olarak kalırken tanınan olguları çıkarır."""
    facts: list[dict[str, Any]] = []
    documented: set[tuple[str, str]] = set()
    for row in rows:
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        evidence = _evidence(row)
        source_key = (evidence["source_file"], evidence["document_category"])
        if source_key not in documented:
            documented.add(source_key)
            facts.append(_fact("document_profile", {
                "document_category": evidence["document_category"],
                "source_type": evidence["source_type"],
            }, row, text[:250]))

        for match in _COURT_DECISION.finditer(text):
            date, case_number, decision_number = match.groups()
            facts.append(_fact("constitutional_court_annulment", {
                "decision_date": date,
                "case_number": case_number,
                "decision_number": decision_number,
                "outcome": "iptal",
                "related_instruments": sorted({f"KHK-{number}" for number in _KHK_REFERENCE.findall(text)}),
            }, row, match.group(0)))
        for match in _DURATION.finditer(text):
            amount, unit = match.groups()
            facts.append(_fact("legal_time_limit", {
                "amount": int(amount), "unit": unit.casefold(),
            }, row, match.group(0)))
        references = sorted({match.group(1) for match in _LAW_REFERENCE.finditer(text)})
        if references:
            facts.append(_fact("cross_law_reference", {"referenced_law_numbers": references}, row, text))
        for match in _AMENDMENT.finditer(text):
            facts.append(_fact("amendment_note", {"note": match.group(1).strip()}, row, match.group(0)))
    return facts


def _ledger_facts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    facts: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("records", []), start=1):
        values = {
            "target_law_number": str(item.get("hedef_kanun_no") or "unknown"),
            "target_articles": list(item.get("hedef_madde_nolari") or []),
            "amending_instrument": str(item.get("degistiren_duzenleme_metni") or ""),
            "amending_instrument_number": str(item.get("degistiren_duzenleme_no") or ""),
            "effective_date": str(item.get("yururluk_tarihi") or ""),
            "effective_date_raw": str(item.get("yururluk_tarihi_ham") or ""),
        }
        source = {
            "chunk_id": f"ledger:{values['target_law_number']}:{index}",
            "source_file": str(item.get("kaynak_dosya") or "unknown"),
            "source_type": "amendment_ledger",
            "document_category": "law_amendment_table",
            "law_number": values["target_law_number"],
            "article_no": ", ".join(values["target_articles"]) or "unknown",
            "page_start": item.get("kaynak_sayfa"),
            "page_end": item.get("kaynak_sayfa"),
        }
        facts.append({
            "fact_id": _record_id("amendment_effective_date", values, source["chunk_id"]),
            "fact_type": "amendment_effective_date",
            "values": values,
            "evidence": source,
            "evidence_text": str(item.get("ham_kanit") or ""),
            "confidence": 1.0,
            "extraction_method": "amendment_ledger",
        })
    return facts


def build_facts_registry(
    rows: Iterable[dict[str, Any]], *, output_path: Path | None = None, ledger_path: Path | None = None
) -> dict[str, Any]:
    """Tüm vector kaynaklarını tekrar embed etmeden facts_registry.json üretir."""
    output = output_path or facts_registry_config.output_path
    facts = extract_facts(rows)
    facts.extend(_ledger_facts(ledger_path or amendment_ledger_config.output_path))
    unique = {fact["fact_id"]: fact for fact in facts}
    ordered = list(unique.values())
    payload = {
        "schema_version": "facts-registry-v1",
        "records": ordered,
        "summary": dict(sorted(Counter(item["fact_type"] for item in ordered).items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return {"records": len(ordered), "summary": payload["summary"], "path": str(output)}
