"""Public ingestion API, loaded lazily to support ``python -m`` execution."""

__all__ = [
    "IngestionReport",
    "build_vector_database",
    "ingest_directory",
    "ingest_documents",
    "ingest_file",
    "reindex_file",
    "upload_file",
]


def __getattr__(name: str):
    """Avoid importing ``pipeline`` before runpy executes it as a module."""
    if name not in __all__:
        raise AttributeError(name)
    if name == "upload_file":
        from RAG.ingestion.uploader import upload_file

        return upload_file
    from RAG.ingestion import pipeline

    return getattr(pipeline, name)
