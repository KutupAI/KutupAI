"""
embedding_config.py
--------------------
HuggingFace embedding ayarları.

Değerler ``RAG/configuration/rag_config.yaml`` dosyasından okunur; burada
yalnız CUDA/CPU seçim mantığı tutulur.
"""

from dataclasses import dataclass
import torch

from RAG.configuration.rag_config_loader import embedding_runtime_config, runtime_config


def _get_best_device() -> str:
    """
    Merkezi ``runtime.device`` ayarına uyar; CUDA yoksa güvenle CPU'ya döner.
    """
    if runtime_config.device.startswith("cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

    
@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = embedding_runtime_config.model_name
    device: str = _get_best_device()
    embedding_dim: int = embedding_runtime_config.embedding_dim
    normalize_embeddings: bool = embedding_runtime_config.normalize_embeddings
    # GPU belleği yetersizse rag_config.yaml içindeki batch_size_cuda azaltılır.
    batch_size: int = (
        embedding_runtime_config.batch_size_cuda
        if _get_best_device() == "cuda"
        else embedding_runtime_config.batch_size_cpu
    )
    show_progress: bool = embedding_runtime_config.show_progress


embedding_config = EmbeddingConfig()
