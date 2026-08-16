"""
update_index.py
------------------
إعادة فهرسة ملف قانوني محدد عند تعديله (بدلاً من إعادة فهرسة كل شيء
من الصفر). المرحلة الأولى: تنفيذ بسيط لكن صحيح - حذف كل الـ chunks
القديمة المرتبطة بهذا الملف، ثم إعادة فهرسته من جديد بالكامل.
"""

from pathlib import Path

from RAG.indexing.indexer import index_file
from RAG.vector_store.chroma_store import get_vector_store


def reindex_file(file_path: Path) -> int:
    """
    إعادة فهرسة ملف واحد: حذف الـ chunks القديمة المرتبطة به ثم إعادة
    إدخاله بالكامل. يُستخدم عند تعديل نص قانوني موجود مسبقًا.

    Args:
        file_path: مسار الملف الذي تم تعديله.

    Returns:
        عدد الـ chunks الجديدة بعد إعادة الفهرسة.
    """
    store = get_vector_store()

    # حذف كل الـ chunks القديمة المرتبطة بهذا الملف تحديدًا عبر الـ metadata
    store.delete(where={"source_file": file_path.name})

    # إعادة الفهرسة الكاملة لنفس الملف بعد التعديل
    new_count = index_file(file_path)
    return new_count


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("الاستخدام: python update_index.py <path_to_modified_file>")
        sys.exit(1)

    target_file = Path(sys.argv[1])
    if not target_file.exists():
        print(f"الملف غير موجود: {target_file}")
        sys.exit(1)

    count = reindex_file(target_file)
    print(f"تمت إعادة فهرسة '{target_file.name}' بـ {count} chunk جديد.")
