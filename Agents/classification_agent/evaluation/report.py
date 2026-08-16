"""
report.py
-----------
Assembles the final deliverables report -- task doc section 11's checklist,
item by item:

  1. 18 class icin veri dagilimi tablosu           -> dataset/distribution.py
  2. Train/validation/test veri yapisi              -> dataset/splitter.py
  3. Secilen Qwen modeli ve inference/quant ayarlari -> config snapshot
  4. Classification Agent calisan pipeline'i        -> (n/a here, see agent.py)
  5. JSON output schema                             -> models.py schema sample
  6. Confidence ve needs_review mekanizmasi          -> config snapshot
  7. Macro-F1, Precision, Recall, Weighted-F1        -> evaluation/metrics.py
  8. Confusion Matrix                                -> evaluation/metrics.py
  9. Per-class sonuclar                              -> evaluation/metrics.py
  10. Zor test senaryolarinin sonuclari              -> evaluation/hard_cases.py
  11. Latency ve VRAM olcumleri                      -> metrics.latency + vram_mb arg
  12. En iyi konfigurasyonun neden secildigine dair kisa sonuc -> caller-supplied text

This module only formats what other modules computed -- it does not invent
numbers. Every section is either filled from real inputs or explicitly
marked "PENDING" so a half-finished report is never mistaken for a
finished one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Agents.classification_agent.dataset.distribution import class_distribution, to_markdown_table
from Agents.classification_agent.dataset.schema import LabeledDocument


def _section(title: str, body: str | None) -> str:
    return f"## {title}\n\n{body.strip() if body else '_PENDING -- veri/sonuç henüz mevcut değil._'}\n"


def build_report(
    *,
    all_records: list[LabeledDocument] | None = None,
    splits: dict[str, list[LabeledDocument]] | None = None,
    model_config_snapshot: dict[str, Any] | None = None,
    evaluation_metrics: dict[str, Any] | None = None,
    hard_case_results: dict[str, Any] | None = None,
    ablation_results: dict[str, Any] | None = None,
    vram_mb: float | None = None,
    final_choice_rationale: str | None = None,
) -> str:
    parts: list[str] = ["# Classification Agent -- Deliverables Report\n"]

    # 1. class distribution
    dist_body = None
    if all_records:
        dist_body = to_markdown_table(class_distribution(all_records))
    parts.append(_section("1. 18 Class için Veri Dağılımı", dist_body))

    # 2. split structure
    split_body = None
    if splits:
        split_body = "\n".join(f"- **{name}**: {len(records)} belge" for name, records in splits.items())
    parts.append(_section("2. Train/Validation/Test Veri Yapısı", split_body))

    # 3 + 6. model + threshold config
    config_body = None
    if model_config_snapshot:
        config_body = "\n".join(f"- `{k}`: {v}" for k, v in model_config_snapshot.items())
    parts.append(_section("3. Seçilen Model ve Inference Ayarları (Confidence/needs_review dahil)", config_body))

    # 5. JSON schema
    schema_sample = (
        "```json\n"
        "{\n"
        '  "document_id": "DOC-001",\n'
        '  "document_type": "dilekce",\n'
        '  "confidence": 0.94,\n'
        '  "alternatives": [{"type": "talep_yazisi", "confidence": 0.04}],\n'
        '  "status": "success"\n'
        "}\n"
        "```"
    )
    parts.append(_section("5. JSON Output Schema", schema_sample))

    # 7-9. metrics + confusion matrix + per-class
    metrics_body = None
    if evaluation_metrics:
        m = evaluation_metrics
        metrics_body = (
            f"- Accuracy: **{m.get('accuracy')}**\n"
            f"- Macro-F1: **{m.get('macro_f1')}**\n"
            f"- Weighted-F1: **{m.get('weighted_f1')}**\n\n"
            f"### Per-class\n\n"
            + "\n".join(
                f"- `{label}`: P={vals['precision']} R={vals['recall']} F1={vals['f1']} (n={vals['support']})"
                for label, vals in (m.get("per_class") or {}).items()
            )
        )
    parts.append(_section("7-9. Macro-F1 / Precision / Recall / Weighted-F1 / Per-class / Confusion Matrix", metrics_body))

    # 10. hard cases
    hard_body = None
    if hard_case_results:
        hard_body = "\n".join(
            f"- `{code}` ({info['description']}): n={info['n']}, accuracy={info['accuracy']}"
            for code, info in hard_case_results.items()
        )
    parts.append(_section("10. Zor Test Senaryolarının Sonuçları", hard_body))

    # 10b (task §10). ablation
    ablation_body = None
    if ablation_results:
        rows = []
        for name, data in ablation_results.items():
            met = data["metrics"]
            rows.append(f"- **{name}** ({data['description']}): Macro-F1={met['macro_f1']}, latency p50={met['latency']['p50_ms']}ms")
        ablation_body = "\n".join(rows)
    parts.append(_section("Ablation / Karşılaştırma Testleri", ablation_body))

    # 11. latency + VRAM
    latency_body = None
    if evaluation_metrics and evaluation_metrics.get("latency"):
        lat = evaluation_metrics["latency"]
        latency_body = (
            f"- mean: {lat['mean_ms']}ms, p50: {lat['p50_ms']}ms, p95: {lat['p95_ms']}ms, max: {lat['max_ms']}ms (n={lat['n']})\n"
            f"- VRAM: {vram_mb if vram_mb is not None else 'PENDING -- ölçülmedi'}"
        )
    parts.append(_section("11. Latency ve VRAM Ölçümleri", latency_body))

    # 12. rationale
    parts.append(_section("12. En İyi Konfigürasyonun Seçilme Gerekçesi", final_choice_rationale))

    return "\n".join(parts)


def write_report(report_markdown: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_markdown, encoding="utf-8")
    return output_path
