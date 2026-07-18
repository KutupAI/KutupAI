"""
context_formatter.py
-----------------------
تنسيق نتائج الاسترجاع (SearchResult) كنص context جاهز للاستخدام من
قبل writer_agent / rag_agent، مع ذكر مصدر كل مقطع بوضوح.

قاعدة صلبة:
هذا الملف لا يضيف أي تعليمات صياغة أو System Prompt - فقط تنسيق
المعلومات المسترجعة. تعليمات الصياغة (كيف يكتب الرد الرسمي) هي مسؤولية
writer_agent/prompts.py حصرًا، وليست جزءًا من طبقة RAG.
"""

from RAG.vector_store.vector_store_interface import SearchResult


def format_context(results: list[SearchResult]) -> str:
    """
    تحويل قائمة نتائج الاسترجاع إلى نص واحد منظّم، كل مقطع مسبوق
    بمصدره (اسم القانون + رقم المادة) لتسهيل الاستشهاد لاحقًا.

    Args:
        results: نتائج retriever.retrieve().

    Returns:
        نص context جاهز، أو نص يوضح عدم وجود نتائج إذا كانت القائمة فارغة.
    """
    if not results:
        return "لا توجد نصوص قانونية ذات صلة تم العثور عليها."

    formatted_sections = []
    for i, result in enumerate(results, start=1):
        meta = result["metadata"]
        source_label = f"[{meta.get('law_name', 'غير معروف')} - المادة {meta.get('article_number', 'غير محددة')}]"
        section = f"{i}. {source_label}\n{result['text']}"
        formatted_sections.append(section)

    return "\n\n".join(formatted_sections)


def extract_sources(results: list[SearchResult]) -> list[dict]:
    """
    استخراج قائمة مصادر مختصرة (بدون النص الكامل) - مفيدة لعرضها بشكل
    منفصل بواجهة المستخدم أو لتسجيلها في الـ logs.

    Args:
        results: نتائج retriever.retrieve().

    Returns:
        قائمة قواميس تحتوي: law_name, article_number, effective_date, score.
    """
    sources = []
    for result in results:
        meta = result["metadata"]
        sources.append(
            {
                "law_name": meta.get("law_name", "unknown"),
                "article_number": meta.get("article_number", "unknown"),
                "effective_date": meta.get("effective_date", "unknown"),
                "score": result["score"],
            }
        )
    return sources
