"""Interactively query the existing legal RAG index without rebuilding it.

Examples:
    python -m RAG.scripts.query_retrieval
    python -m RAG.scripts.query_retrieval "CMK 100. madde neyi düzenler?"
"""

from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter
from typing import Any, Dict


def _milliseconds(value: object) -> str:
    """Tanı ekranındaki süreleri okunabilir ve tek biçimli gösterir."""
    try:
        return f"{float(value):,.1f} ms"
    except (TypeError, ValueError):
        return "-"


def _print_debug(trace: Dict[str, Any], plan, config) -> None:
    """Ayar dosyası izin verirse kısa ve kullanıcı odaklı retrieval özeti verir."""
    if not config.retrieval_debug:
        return
    techniques = []
    mode = trace.get("mode")
    if mode == "hybrid":
        techniques.extend(["Vector Search", "BM25", "RRF"])
    elif mode == "bm25":
        techniques.append("BM25")
    else:
        techniques.append("Vector Search")
    if trace.get("prf_applied"):
        techniques.append("PRF")
    if trace.get("graph_enabled"):
        techniques.append("Graph-RAG")
    if trace.get("reranker_enabled"):
        techniques.append("Cross-Encoder Reranker")

    original = str(trace.get("input_query") or "").strip()
    variants = [str(item).strip() for item in (trace.get("query_variants") or []) if str(item).strip()]
    changed_variants = [item for item in variants if item.casefold() != original.casefold()]

    print("\n--- RETRIEVAL ÖZETİ ---")
    print(f"Yol: {plan.name}")
    print(f"Kullanılan teknikler: {', '.join(techniques)}")
    if changed_variants:
        print("Query Transform:")
        for variant in changed_variants:
            print(f"  {original} -> {variant}")
    else:
        print("Query Transform: Değişiklik yapılmadı.")
    if config.show_candidate_details:
        print(
            "Aday sayıları: "
            f"ilk={trace.get('initial_candidates_after_dedup', 0)} | "
            f"PRF sonrası={trace.get('candidates_after_prf', 0)} | "
            f"Graph sonrası={trace.get('candidates_after_graph', 0)} | "
            f"nihai={trace.get('final_result_count', 0)}"
        )
        for index, item in enumerate(trace.get("search_variants") or [], start=1):
            print(
                f"  Varyant {index}: mode={item.get('mode')} | filtre={item.get('metadata_filter') or '-'} | "
                f"vector={item.get('vector_candidates', 0)} | BM25={item.get('bm25_candidates', 0)} | "
                f"RRF={item.get('fused_candidates', 0)}"
            )
    if config.show_stage_timings:
        timings = [f"Query Transform {_milliseconds(trace.get('query_transform_ms'))}", f"Arama {_milliseconds(trace.get('initial_search_ms'))}"]
        if trace.get("prf_applied"):
            timings.append(f"PRF {_milliseconds(trace.get('prf_ms'))}")
        if trace.get("graph_enabled"):
            timings.append(f"Graph-RAG {_milliseconds(trace.get('graph_ms'))}")
        if trace.get("reranker_enabled"):
            timings.append(f"Reranker {_milliseconds(trace.get('reranker_ms'))}")
        timings.append(f"Toplam {_milliseconds(trace.get('total_retrieval_ms'))}")
        print("Süreler: " + " | ".join(timings))


def main() -> None:
    # PowerShell terminals can still inherit a legacy Windows code page.
    # Türkçe metin ve sonuç işaretleri desteklenen her terminalde okunur kalır.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="Optional Turkish legal question; omit for interactive mode.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["auto", "vector", "bm25", "hybrid"],
        default="auto",
        help="auto uses Query Router; other values force one retrieval mode for experiments.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking for the faster retrieval-only mode.",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Force BGE cross-encoder reranking even if the router would skip it.",
    )
    parser.add_argument("--prf", action="store_true", help="Force PRF experiment mode.")
    parser.add_argument("--no-prf", action="store_true", help="Disable PRF even if the router selects it.")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Show detailed router, candidate and timing information.")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Hide detailed router and timing information.")
    parser.set_defaults(debug=None)
    parser.add_argument(
        "--graph-rag", choices=["auto", "on", "full", "off"], default="auto",
        help="Legal graph expansion; full also follows verified cross-law article references.",
    )
    parser.add_argument("--source-type", choices=["laws", "regulations", "amendments", "internal_docs"])
    parser.add_argument(
        "--query-transform",
        action="store_true",
        help="Enable the local Qwen2.5-1.5B query transformer; falls back safely if unavailable.",
    )
    args = parser.parse_args()

    # Retriever import edildiğinde config okunur; opsiyonel doğruluk modu önce
    # seçilir. Normal komut hızlı kalır.
    if args.query_transform:
        os.environ["RAG_QUERY_TRANSFORM_ENABLED"] = "1"
    from RAG.configuration.rag_config_loader import observability_config
    from RAG.retriever.query_router import choose_query_plan
    from RAG.retriever.retriever import retrieve
    from RAG.retriever.query_intent import service_lookup_notice

    def ask(query: str) -> None:
        plan = choose_query_plan(query)
        mode = plan.mode if args.mode == "auto" else args.mode
        use_prf = False if args.no_prf else (True if args.prf else plan.use_prf)
        use_reranker = False if args.no_reranker else (True if args.reranker else plan.use_reranker)
        graph_mode = {"auto": plan.use_graph, "on": True, "full": "full", "off": False}[args.graph_rag]
        label = f"{mode.upper()} RETRIEVAL"
        label += " + QUERY TRANSFORM" if args.query_transform else " (FAST PRECISION)"
        if use_prf:
            label += " + PRF"
        if use_reranker:
            label += " + RERANKER"
        label += f" | GRAPH-RAG: {args.graph_rag.upper()}"
        print(f"\n===== {label} =====")
        trace: Dict[str, Any] = {}
        started = perf_counter()
        results = retrieve(
            query=query,
            top_k=args.top_k,
            mode=mode,
            use_prf=use_prf,
            use_graph=graph_mode,
            use_reranker=use_reranker,
            where={"source_type": args.source_type} if args.source_type else None,
            trace=trace,
        )
        trace.setdefault("total_retrieval_ms", round((perf_counter() - started) * 1000, 3))
        debug_config = observability_config
        if args.debug is not None:
            debug_config = type(observability_config)(
                retrieval_debug=args.debug,
                show_stage_timings=observability_config.show_stage_timings,
                show_query_details=observability_config.show_query_details,
                show_candidate_details=observability_config.show_candidate_details,
                show_result_metadata=observability_config.show_result_metadata,
            )
        if not results:
            _print_debug(trace, plan, debug_config)
            print("\nSonuç bulunamadı.")
            return
        notice = service_lookup_notice(query, results)
        if notice:
            print(f"\n⚠️ {notice}")

        print(f"\n✅ {len(results)} adet sonuç bulundu.")
        print("\n--- FINAL RETRIEVED CHUNKS ---")
        print("=" * 60)
        for rank, result in enumerate(results, start=1):
            meta = result["metadata"]
            law_number = meta.get("law_number") or meta.get("law_no") or "Bilinmiyor"
            article_no = meta.get("article_no") or meta.get("article_number") or "Genel"
            source_file = meta.get("source_file") or meta.get("source") or "Bilinmiyor"
            page_start = meta.get("page_start") or meta.get("page") or "?"
            text = result["text"].replace("\n", " ").strip()
            metadata_text = (
                f"\n📄 RESULT {rank}\n"
                f"⭐ Score: {result['score']:.4f}\n"
                f"🆔 Chunk ID: {meta.get('chunk_id') or result['id']}\n"
                f"📌 Kanun No: {law_number}\n"
                f"📌 Madde No: {article_no}\n"
                f"📌 Madde Tipi: {meta.get('article_type', 'Bilinmiyor')}\n"
                f"📌 Kaynak Dosya: {source_file}\n"
                f"📌 Sayfa: {page_start}\n"
            ) if debug_config.show_result_metadata else f"\n📄 RESULT {rank}\n⭐ Score: {result['score']:.4f}\n"
            print(
                metadata_text
                +
                "\n📝 CONTENT (İlk 500 karakter):\n"
                + "-" * 60
                + f"\n{text[:500]}{'...' if len(text) > 500 else ''}\n"
                + "=" * 60
            )
        _print_debug(trace, plan, debug_config)

    if args.query:
        ask(args.query)
        return

    print("Hukuki RAG interaktif sorgu modu. Çıkmak için: exit / çıkış / q")
    while True:
        try:
            query = input("\nSorunuz> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış yapıldı.")
            break
        if query.casefold() in {"exit", "cikis", "çıkış", "q", "quit"}:
            print("Çıkış yapıldı.")
            break
        if query:
            ask(query)


if __name__ == "__main__":
    main()
