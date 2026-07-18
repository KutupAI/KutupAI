"""
embedding_config.py
--------------------
إعدادات نموذج الـ Embeddings (BGE-M3) المستخدم في طبقة RAG.

هذا الملف لا يحتوي على أي منطق تنفيذي (لا تحميل نموذج، لا استدعاءات) -
فقط قيم إعداد ثابتة يستوردها embeddings/embedding_model.py.

قابلية الاستبدال:
لتغيير نموذج الـ Embeddings لاحقًا، يكفي تعديل MODEL_NAME و EMBEDDING_DIM
هنا فقط، دون المساس بـ embedding_model.py أو أي طبقة أخرى.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    # اسم النموذج كما يُحمَّل من HuggingFace / sentence-transformers
    model_name: str = "BAAI/bge-m3"

    # الجهاز المستخدم للتشغيل - CPU فقط في هذه المرحلة (بدون CUDA)
    device: str = "cpu"

    # أبعاد المتجه الناتج عن BGE-M3 (ثابتة حسب النموذج)
    embedding_dim: int = 1024

    # الحد الأقصى لعدد التوكنز التي يقبلها النموذج لكل نص إدخال
    max_input_tokens: int = 8192

    # هل يتم تطبيع المتجهات (Normalization) بعد التوليد
    # مهم لتوافق البحث بالتشابه الكوني (Cosine Similarity) مع ChromaDB
    normalize_embeddings: bool = True

    # حجم الدفعة (batch) عند توليد embeddings لعدة نصوص دفعة واحدة
    batch_size: int = 32


# نسخة وحيدة (singleton) يتم استيرادها في باقي الملفات
embedding_config = EmbeddingConfig()
