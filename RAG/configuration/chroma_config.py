"""
chroma_config.py
------------------
إعدادات ChromaDB المستخدم كتطبيق حالي لـ vector_store_interface.py.

مهم:
هذا الملف هو المكان الوحيد الذي يجب أن يحتوي على تفاصيل ChromaDB
(المسار، اسم الـ collection، نوع المسافة). أي كود خارج vector_store/
يجب ألا يعرف بوجود ChromaDB إطلاقًا - فقط يتعامل مع
vector_store_interface.py.

قابلية الاستبدال:
عند تغيير قاعدة المتجهات مستقبلًا (مثال: Qdrant)، هذا الملف يُستبدل
بملف إعداد مكافئ، ويُكتب تطبيق جديد في vector_store/، دون التأثير
على retriever.py أو أي Agent.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChromaConfig:
    # مسار التخزين الدائم (persistent storage) على القرص
    persist_directory: str = "RAG/documents/.chroma_db"

    # اسم الـ collection الذي سيُخزَّن فيه كل القوانين والتشريعات
    collection_name: str = "legal_documents"

    # نوع مقياس المسافة المستخدم عند البحث بالتشابه
    # الخيارات المتاحة في ChromaDB: "cosine" | "l2" | "ip"
    distance_metric: str = "cosine"

    # هل يُعاد إنشاء الـ collection من الصفر إذا كانت موجودة مسبقًا
    # (يُستخدم فقط عند إعادة الفهرسة الكاملة - وليس في التشغيل العادي)
    reset_on_full_reindex: bool = False


# نسخة وحيدة (singleton) يتم استيرادها في chroma_store.py فقط
chroma_config = ChromaConfig()
