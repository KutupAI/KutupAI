"""
distribution.py
------------------
Class distribution table -- task doc section 6's first required step
("Her sinifta kac ornek oldugunu gosteren tablo hazirlanmali") and part of
the section 11 deliverables ("18 class icin veri dagilimi tablosu").

Also flags minority classes, since section 6 explicitly calls out that
under-represented classes can bias the model toward the majority class.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from Agents.classification_agent.dataset.loader import load_manifest
from Agents.classification_agent.dataset.schema import LabeledDocument
from Agents.classification_agent.taxonomy import DOCUMENT_CLASSES

# Below this count, a class is flagged as needing weighting/oversampling/
# augmentation attention (section 6). Not a hard rule -- a starting signal.
MINORITY_THRESHOLD = 20


def class_distribution(records: list[LabeledDocument]) -> dict[str, int]:
    """Counts per taxonomy code, including zero for classes with no
    examples yet -- a class with 0 real examples is exactly what section 5
    says to revisit/merge/drop, and it should be visible in the table, not
    silently missing from it.
    """
    counts = Counter(r.label for r in records if r.label)
    return {c.code: counts.get(c.code, 0) for c in DOCUMENT_CLASSES}


def to_markdown_table(distribution: dict[str, int]) -> str:
    total = sum(distribution.values())
    code_to_name = {c.code: c.tr_name for c in DOCUMENT_CLASSES}

    lines = ["| # | code | Türkçe adı | adet | oran | uyarı |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(DOCUMENT_CLASSES, start=1):
        count = distribution.get(c.code, 0)
        pct = f"{(count / total * 100):.1f}%" if total else "0.0%"
        warning = "AZ ÖRNEK — bkz. §6" if count < MINORITY_THRESHOLD else ""
        lines.append(f"| {i} | `{c.code}` | {code_to_name[c.code]} | {count} | {pct} | {warning} |")
    lines.append(f"| | **TOPLAM** | | **{total}** | 100% | |")
    return "\n".join(lines)


def minority_classes(distribution: dict[str, int], threshold: int = MINORITY_THRESHOLD) -> list[str]:
    return [code for code, count in distribution.items() if count < threshold]


def zero_support_classes(distribution: dict[str, int]) -> list[str]:
    """Classes with literally no real examples -- section 5's "gerçek veriyle
    desteklenmeyen sınıflar" that must be reworked, not force-kept."""
    return [code for code, count in distribution.items() if count == 0]


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print class distribution table from a labeling manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None, help="Optional path to save the markdown table.")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    dist = class_distribution(records)
    table = to_markdown_table(dist)

    print(table)
    print()
    zero = zero_support_classes(dist)
    minority = [c for c in minority_classes(dist) if c not in zero]
    if zero:
        print(f"UYARI -- gerçek örneği olmayan sınıflar (§5'e göre gözden geçir): {zero}")
    if minority:
        print(f"UYARI -- az örnekli sınıflar (§6, weighting/oversampling düşün): {minority}")

    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    _cli()
