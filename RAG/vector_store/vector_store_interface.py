"""
vector_store_interface.py
----------------------------
العقد الثابت (Interface) الذي يعتمد عليه باقي طبقة RAG.

القاعدة الصلبة:
لا يُسمح لأي ملف خارج vector_store/ أن يستورد ChromaDB مباشرة.
كل تعامل مع مخزن المتجهات يجب أن يمر عبر هذا العقد.

قابلية الاستبدال:
لتغيير قاعدة المتجهات لاحقًا (مثال: من ChromaDB إلى Qdrant)، يكفي كتابة
كلاس جديد ينفّذ VectorStoreInterface، دون تعديل retriever.py أو indexer.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict


class SearchResult(TypedDict):
    """شكل موحّد لنتيجة بحث واحدة، بغض النظر عن التطبيق الفعلي للمخزن."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # كلما اقترب من 1 كان أكثر تشابهًا (حسب مقياس المسافة المستخدم)


class VectorStoreInterface(ABC):
    """العقد الذي يجب أن ينفّذه أي تطبيق لمخزن المتجهات."""

    @abstractmethod
    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        إضافة مجموعة من الـ chunks إلى المخزن.

        Args:
            ids: معرّف فريد لكل chunk.
            texts: النص الخام لكل chunk (يُخزَّن للرجوع إليه لاحقًا).
            embeddings: المتجه المقابل لكل نص (بنفس الترتيب).
            metadatas: بيانات وصفية لكل chunk (اسم القانون، رقم المادة، ...).
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        البحث عن أقرب top_k نتائج لمتجه الاستعلام.

        Args:
            query_embedding: متجه سؤال المستخدم.
            top_k: عدد النتائج المطلوبة.
            where: فلترة اختيارية على الـ metadata (مثال: {"law_name": "..."})

        Returns:
            قائمة SearchResult مرتبة تنازليًا حسب درجة التشابه.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, where: Dict[str, Any]) -> None:
        """
        حذف كل الـ chunks المطابقة لشرط معيّن (يُستخدم عند إعادة الفهرسة).

        Args:
            where: شرط الحذف (مثال: {"source_file": "kanun_5651.txt"})
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """إرجاع عدد الـ chunks المخزّنة حاليًا بالكامل."""
        raise NotImplementedError
