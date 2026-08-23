"""Input/output contracts for summary_agent (RAG in → summary out)."""

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
    """Matches RAG/client output; also accepted via state['rag_result']."""

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
    """Standalone envelope; pipeline flattens this into state['summary']."""

    success: bool
    data: Optional[SummaryData] = None
    error: Optional[dict] = None
