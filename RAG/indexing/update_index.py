"""
update_index.py
------------------
Değişen tek bir hukukî dosyayı, tüm corpus'u sıfırdan kurmadan yeniden indeksler.
Önce dosyaya bağlı eski chunk'ları siler, ardından aynı dosyayı bütünüyle ekler.
"""

from pathlib import Path

from RAG.indexing.indexer import index_file
from RAG.vector_store.chroma_store import get_vector_store


def reindex_file(file_path: Path) -> int:
    """
    Tek dosyanın eski chunk'larını siler ve güncel metni tam olarak yeniden ekler.

    Args:
        file_path: Değiştirilen dosyanın yolu.

    Returns:
        Yeniden indeksleme sonrası oluşan yeni chunk sayısı.
    """
    store = get_vector_store()

    # Bu dosyaya ait eski chunk'lar metadata üzerinden hedefli biçimde silinir.
    store.delete(where={"source_file": file_path.name})

    # Değişen aynı dosya tam olarak yeniden indekslenir.
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
