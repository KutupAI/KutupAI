"""İnteraktif, çok turlu ve kaynaklı hukukî RAG sohbeti.

Bu dosya benchmark değildir. Kullanıcının sorusunu Qwen ile yanıtlar, retrieval
planını, kaynakları ve süreleri gösterir. Takip sorusu önceki konuyla ilişkiliyse
eski soruyu tekrar aramaz; yalnız yeni bilgiyi son kanun kapsamında getirir.

Çalıştırma:
    python Tests/RAG/run_llm_evaluation.py
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path


# Dosya doğrudan çalıştırıldığında proje paketi kökten bulunur.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows kod sayfasından bağımsız Türkçe çıktı üretir.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _print_result(result) -> None:
    """Cevap, konuşma bağı, kaynaklar ve performans bilgisini düzenli gösterir."""
    answer = result.answer
    print("\n===== HUKUKİ RAG SOHBET =====\n")
    # Retrieval katmanı özgün soruyu korur; varsa yalnız arama için üretilen
    # yazım düzeltilmiş/yeniden ifade edilmiş biçimleri kullanıcıya açıkça gösterilir.
    if result.retrieval_query:
        from RAG.retriever.query_transform import transform_query

        transformed = transform_query(result.retrieval_query)
        extra_queries = transformed.queries[1:]
        if extra_queries:
            llm_queries = set(transformed.llm_queries or [])
            deterministic_queries = [query for query in extra_queries if query not in llm_queries]
            if deterministic_queries:
                print("🔄 Arama için düzeltilen sorgu:")
                for query in deterministic_queries:
                    print(f"   - {query}")
            if llm_queries:
                print("🤖 Query Transform LLM'in ürettiği alternatif sorgular:")
                for query in transformed.llm_queries or []:
                    print(f"   - {query}")
            if not deterministic_queries and not llm_queries:
                print("🔄 Arama için eklenen sorgu:")
                for query in extra_queries:
                    print(f"   - {query}")
            print()
    if result.related_to_previous and result.previous_turn:
        previous = result.previous_turn
        law = previous.primary_law_name or previous.primary_law_number or "önceki hukukî konu"
        print(f"🔗 Önceki konuyla bağlantı kuruldu: {law}")
        print("   Eski cevap yeniden aranmadı; yalnız yeni soru için ek kanıt getirildi.\n")
    elif result.memory_used:
        print("🧠 Son konuşma bağlamı, soru anlamını korumak için LLM'e iletildi.\n")

    print(answer.answer)
    if answer.retrieval_plan:
        print(f"\n--- SEÇİLEN YOL: {answer.retrieval_plan} ---")
        print(answer.retrieval_plan_reason)
    if answer.citations:
        print("\n--- KAYNAKLAR ---")
        print(answer.citations)
    print(
        f"\nGrounded={answer.grounded} | Cache={answer.cache_hit} | "
        f"Retrieval={answer.retrieval_ms:.0f} ms | Generation={answer.generation_ms:.0f} ms | "
        f"Total={answer.total_ms:.0f} ms"
    )
    if answer.refusal_reason:
        print(f"Neden: {answer.refusal_reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5, help="Her soru için en fazla kaynak sayısı.")
    parser.add_argument("--no-cache", action="store_true", help="Disk üzerindeki semantic cache'i kullanmaz.")
    parser.add_argument(
        "--query-transform-llm",
        action="store_true",
        help="8081'deki ayrı Qwen servisiyle ek sorgu varyantları üretir.",
    )
    args = parser.parse_args()

    # Ayar modülü ilk import edildiğinde okunur. Geçici komut satırı tercihi
    # bu nedenle RAG bileşenleri import edilmeden önce ortam değişkenine yazılır.
    if args.query_transform_llm:
        os.environ["RAG_QUERY_TRANSFORM_USE_LLM"] = "true"

    from RAG.agent.conversation import LegalConversation
    from RAG.configuration.rag_config_loader import query_transform_config

    conversation = LegalConversation()
    print("Hukuki RAG çok turlu sohbet modu.")
    print("Çıkış: exit / q | Yeni bağımsız konu: clear")
    if query_transform_config.enabled:
        mode = "LLM (Qwen)" if query_transform_config.use_llm else "hızlı kural tabanlı"
        print(f"Query Transform: açık — {mode}")
        if query_transform_config.use_llm:
            try:
                host_port = query_transform_config.base_url.split("//", 1)[-1].split("/", 1)[0]
                host, port = host_port.rsplit(":", 1)
                with socket.create_connection((host, int(port)), timeout=1):
                    pass
            except (OSError, ValueError):
                print("Uyarı: Query Transform LLM servisine ulaşılamıyor; yalnız özgün/kurallı sorgu kullanılır.")
    while True:
        try:
            question = input("\nSorunuz> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış yapıldı.")
            return 0
        if question.casefold() in {"exit", "q", "quit", "çıkış", "cikis"}:
            print("Çıkış yapıldı.")
            return 0
        if question.casefold() in {"clear", "temizle", "yeni konu"}:
            conversation.clear()
            print("Oturum belleği temizlendi. Yeni konu için hazır.")
            continue
        if question:
            _print_result(conversation.ask(question, top_k=max(1, args.top_k), use_cache=not args.no_cache))


if __name__ == "__main__":
    raise SystemExit(main())
