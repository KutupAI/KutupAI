"""Interactive grounded legal-answering CLI.

Run: python -m RAG.scripts.ask_legal_agent
"""

from __future__ import annotations

import argparse
import json

from RAG.agent.legal_agent import LegalRagAgent


def _print(answer) -> None:
    print("\n===== HUKUKİ RAG AGENT =====\n")
    print(answer.answer)
    if answer.retrieval_plan:
        print(f"\n--- SEÇİLEN YOL: {answer.retrieval_plan} ---")
        print(answer.retrieval_plan_reason)
    if answer.citations:
        print("\n--- KAYNAKLAR ---")
        print(answer.citations)
    print(
        f"\nGrounded={answer.grounded} | Cache={answer.cache_hit} | "
        f"Retrieval={answer.retrieval_ms:.0f} ms | Generation={answer.generation_ms:.0f} ms | Total={answer.total_ms:.0f} ms"
    )
    if answer.refusal_reason:
        print(f"Neden: {answer.refusal_reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    agent = LegalRagAgent()

    def ask(question: str) -> None:
        answer = agent.answer(question, use_cache=not args.no_cache)
        print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2) if args.json else "") if args.json else _print(answer)

    if args.question:
        ask(args.question)
        return
    print("Hukuki RAG Agent. Sistem soruya göre en uygun retrieval yolunu otomatik seçer. Çıkış: exit / q")
    while True:
        try:
            question = input("\nSorunuz> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.casefold() in {"exit", "q", "quit", "çıkış", "cikis"}:
            break
        if question:
            ask(question)


if __name__ == "__main__":
    main()
