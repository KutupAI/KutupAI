"""
embedding_model.py
--------------------
تحويل النصوص إلى متجهات (vectors) باستخدام BGE-M3.

هذا هو المكان الوحيد الذي يُحمَّل فيه نموذج الـ Embeddings في كامل
طبقة RAG. أي ملف آخر يحتاج تحويل نص لمتجه يستدعي الدوال هنا فقط
ولا يتعامل مع sentence-transformers مباشرة.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from RAG.configuration.embedding_config import embedding_config


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    تحميل النموذج مرة واحدة فقط (Singleton) وإعادة استخدامه.
    lru_cache يضمن عدم إعادة تحميل النموذج من الذاكرة/القرص مع كل استدعاء.
    """
    model = SentenceTransformer(
        embedding_config.model_name,
        device=embedding_config.device,
    )
    return model


def embed_text(text: str) -> List[float]:
    """
    تحويل نص واحد إلى متجه.

    Args:
        text: النص المراد تحويله (chunk قانوني أو query المستخدم).

    Returns:
        قائمة أرقام عشرية (المتجه)، بطول embedding_config.embedding_dim.
    """
    model = _load_model()
    vector = model.encode(
        text,
        normalize_embeddings=embedding_config.normalize_embeddings,
        convert_to_numpy=True,
    )
    return vector.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    تحويل عدة نصوص دفعة واحدة (أسرع من استدعاء embed_text بحلقة).

    Args:
        texts: قائمة نصوص (chunks).

    Returns:
        قائمة متجهات بنفس ترتيب الإدخال.
    """
    if not texts:
        return []

    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=embedding_config.batch_size,
        normalize_embeddings=embedding_config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def get_embedding_dim() -> int:
    """إرجاع أبعاد المتجه المُعرَّفة في الإعدادات (مرجع سريع بدون تحميل النموذج)."""
    return embedding_config.embedding_dim
