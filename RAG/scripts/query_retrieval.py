"""Interactively query the existing legal RAG index without rebuilding it.

Examples:
    python -m RAG.scripts.query_retrieval
    python -m RAG.scripts.query_retrieval "CMK 100. madde neyi düzenler?"
"""

from __future__ import annotations

import argparse
import os
import sys


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
        choices=["vector", "bm25", "hybrid"],
        default="vector",
        help="Default vector mode is the measured best Hit@1 profile; hybrid remains available for recall experiments.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip reranking for the faster retrieval-only mode.",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable BGE cross-encoder reranking (higher latency; use after evaluating your target metric).",
    )
    parser.add_argument("--prf", action="store_true", help="Enable PRF experiment mode.")
    parser.add_argument(
        "--graph-rag", choices=["auto", "on", "full", "off"], default="auto",
        help="Legal graph expansion; full also follows verified cross-law article references.",
    )
    parser.add_argument("--source-type", choices=["laws", "regulations", "amendments", "internal_docs"])
    parser.add_argument(
        "--query-transform",
        action="store_true",
        help="Use the local Qwen2.5-7B query transformer for higher Hit@1; falls back safely if unavailable.",
    )
    args = parser.parse_args()

    # Retriever import edildiğinde config okunur; opsiyonel doğruluk modu önce
    # seçilir. Normal komut hızlı kalır.
    if args.query_transform:
        os.environ["RAG_QUERY_TRANSFORM_ENABLED"] = "1"
    from RAG.retriever.retriever import retrieve
    from RAG.retriever.query_intent import service_lookup_notice

    def ask(query: str) -> None:
        label = f"{args.mode.upper()} RETRIEVAL"
        label += " + QUERY TRANSFORM" if args.query_transform else " (FAST PRECISION)"
        if args.prf:
            label += " + PRF"
        if args.reranker and not args.no_reranker:
            label += " + RERANKER"
        label += f" | GRAPH-RAG: {args.graph_rag.upper()}"
        print(f"\n===== {label} =====")
        results = retrieve(
            query=query,
            top_k=args.top_k,
            mode=args.mode,
            use_prf=args.prf,
            use_graph={"auto": None, "on": True, "full": "full", "off": False}[args.graph_rag],
            use_reranker=args.reranker and not args.no_reranker,
            where={"source_type": args.source_type} if args.source_type else None,
        )
        if not results:
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
            print(
                f"\n📄 RESULT {rank}\n"
                f"⭐ Score: {result['score']:.4f}\n"
                f"🆔 Chunk ID: {meta.get('chunk_id') or result['id']}\n"
                f"📌 Kanun No: {law_number}\n"
                f"📌 Madde No: {article_no}\n"
                f"📌 Madde Tipi: {meta.get('article_type', 'Bilinmiyor')}\n"
                f"📌 Kaynak Dosya: {source_file}\n"
                f"📌 Sayfa: {page_start}\n"
                "\n📝 CONTENT (İlk 500 karakter):\n"
                + "-" * 60
                + f"\n{text[:500]}{'...' if len(text) > 500 else ''}\n"
                + "=" * 60
            )

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
