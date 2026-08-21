"""
Agents/summary_agent/schemas.py

Data contracts for summary_agent:
- Input: the RAG Layer's retrieval result (rag_client output contract).
- Output: the structured summary handed off to writer_agent.
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RAGResultItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    law_number: Optional[str] = None
    article_no: Optional[str] = None
    article_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: Optional[float] = None


class RAGData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    operation: str
    query: str
    document_id: Optional[str] = None
    file_name: Optional[str] = None
    results: List[RAGResultItem] = Field(default_factory=list)


class RAGResult(BaseModel):
    """Mirrors RAG/client/rag_client.py's output contract exactly."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    data: Optional[RAGData] = None
    error: Optional[dict] = None


class SourceRef(BaseModel):
    chunk_id: str
    law_number: Optional[str] = None
    article_no: Optional[str] = None
    article_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class SummaryData(BaseModel):
    operation: str = "summarize_context"
    query: str
    summary: str
    sources: List[SourceRef] = Field(default_factory=list)


class SummaryAgentResult(BaseModel):
    """The agent's output contract, mirroring the RAG contract shape
    so downstream agents (writer_agent) handle both uniformly."""

    success: bool
    data: Optional[SummaryData] = None
    error: Optional[dict] = None
