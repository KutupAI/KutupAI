"""Production-oriented legal-answering layer built on the RAG retriever."""

from RAG.agent.legal_agent import LegalAnswer, LegalRagAgent
from RAG.agent.conversation import ConversationResult, ConversationTurn, LegalConversation

__all__ = ["ConversationResult", "ConversationTurn", "LegalAnswer", "LegalConversation", "LegalRagAgent"]
