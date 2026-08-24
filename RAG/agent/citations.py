"""Citation formatting and deterministic validation for legal answers."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

_CITATION_RE = re.compile(r"\[S(\d+)\]")


def _display_law_name(source: Dict[str, object]) -> str:
    """Dosya adındaki tekrar eden kanun numarasını kullanıcı çıktısından kaldırır."""
    number = str(source.get("law_number") or "").strip()
    name = str(source.get("law_name") or "Bilinmeyen kanun").strip()
    if number and number != "unknown":
        name = re.sub(rf"^{re.escape(number)}[_\s-]*", "", name).strip()
        return f"{number} sayılı {name}"
    return name


def cited_labels(answer: str) -> List[str]:
    return [f"S{number}" for number in _CITATION_RE.findall(answer or "")]


def validate_citations(answer: str, sources: Iterable[Dict[str, object]]) -> Tuple[bool, List[str]]:
    """Return whether the answer cites only supplied context labels."""
    known = {str(source.get("label")) for source in sources}
    observed = cited_labels(answer)
    invalid = sorted(set(observed) - known)
    return bool(observed) and not invalid, invalid


def render_citations(sources: Iterable[Dict[str, object]]) -> str:
    lines: List[str] = []
    for source in sources:
        page = source.get("page_start")
        page_text = f" - Sayfa {page}" if page else ""
        if source.get("source_type") == "reference_docs":
            category = str(source.get("document_category") or "belirsiz")
            lines.append(
                f"[{source.get('label')}] Referans Belge ({category}) - "
                f"{source.get('source_file')}{page_text}"
            )
            continue
        if source.get("source_type") == "legal_facts":
            lines.append(
                f"[{source.get('label')}] Yapılandırılmış Hukukî Olgu - "
                f"{source.get('source_file')}{page_text}"
            )
            continue
        lines.append(
            f"[{source.get('label')}] {_display_law_name(source)} - "
            f"Madde {source.get('article_number')}{page_text}"
        )
    return "\n".join(lines)
