"""Teslim için tam retrieval benchmark çalıştırıcısı.

Bu dosya yalnız geri getirme kalitesini ölçer; LLM çağrısı yapmaz. Her profil
aynı held-out soru seti üzerinde Hit@1/3/5, MRR, gecikme, RAM ve indeks boyutu
üretir. Sonuç JSON dosyası rapora veya CI çıktısına doğrudan eklenebilir.

Örnekler:
    python Tests/RAG/run_retrieval_evaluation.py
    python Tests/RAG/run_retrieval_evaluation.py --profiles precision balanced recall
    python Tests/RAG/run_retrieval_evaluation.py --backend faiss --profiles precision
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Dosya doğrudan çalıştırıldığında proje paketinin import edilebilmesini sağlar.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows'ta ``python dosya.py`` çağrısı eski kod sayfasını devralabilir.
# Türkçe yardım ve rapor metninin her terminalde yazılabilmesi için UTF-8 zorlanır.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Profiller birbirinden bağımsızdır; adil karşılaştırma için yalnız burada
# belirtilen retrieval seçenekleri değişir. Aynı veri, metadata filtresi ve
# relevance ölçütü kullanılır.
PROFILES: dict[str, dict[str, Any]] = {
    "precision": {"mode": "vector", "use_prf": False, "use_reranker": True},
    "balanced": {"mode": "hybrid", "use_prf": False, "use_reranker": True},
    "recall": {"mode": "hybrid", "use_prf": True, "use_reranker": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "RAG" / "evaluation" / "datasets" / "heldout_legal.json",
        help="Kanun ve madde etiketi içeren held-out JSON veri seti.",
    )
    parser.add_argument("--backend", choices=["chroma", "faiss", "turbovec"], default="chroma")
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--top-k", type=int, default=5, help="Her soru için döndürülecek en fazla sonuç sayısı.")
    parser.add_argument("--warmup", type=int, default=5, help="Ölçüm dışı model ve indeks ısıtma sorgusu sayısı.")
    parser.add_argument("--build-faiss", action="store_true", help="FAISS indeksi yoksa Chroma'dan üretir.")
    parser.add_argument("--build-turbovec", action="store_true", help="TurboVec indeksi yoksa Chroma'dan üretir.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "RAG" / "evaluation" / "experiments" / "latest_retrieval_evaluation.json",
        help="Detaylı sonuçların yazılacağı JSON yolu.",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        print(f"HATA: veri seti bulunamadı: {args.dataset}")
        return 2

    from RAG.evaluation.system_benchmark import run_system_benchmark

    reports: dict[str, Any] = {}
    for name in args.profiles:
        settings = PROFILES[name]
        print(f"\n=== RETRIEVAL PROFİLİ: {name.upper()} | {args.backend.upper()} ===")
        report = run_system_benchmark(
            backend=args.backend,
            dataset_path=args.dataset,
            top_k=args.top_k,
            warmup=args.warmup,
            build_turbovec=args.build_turbovec,
            build_faiss=args.build_faiss,
            **settings,
        )
        reports[name] = report
        # Terminalde kısa özet, JSON dosyasında soru bazlı başarısızlık listesi kalır.
        print(json.dumps({key: report[key] for key in ("accuracy", "search_speed", "memory")}, ensure_ascii=False, indent=2))

    payload = {
        "benchmark": "legal_retrieval_delivery_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "backend": args.backend,
        "profiles": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTam retrieval raporu kaydedildi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
