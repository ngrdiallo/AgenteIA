"""
Pydantic models per l'API REST — validazione request/response.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request models ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Richiesta chat con parametri RAG."""
    query: str = Field(..., min_length=1, max_length=5000, description="Domanda utente")
    modalita: str = Field("generale", description="Modalità operativa")
    use_rag: bool = Field(True, description="Usa documenti caricati")
    session_id: Optional[str] = Field(None, description="ID sessione per persistenza chat")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Cronologia conversazione [{'role':'user','content':'...'}]",
    )


class UploadRequest(BaseModel):
    """Metadati upload (il file arriva come multipart)."""
    description: Optional[str] = Field(None, max_length=500)


class VisionRequest(BaseModel):
    """Richiesta analisi immagine."""
    depth: str = Field("standard", description="quick | standard | deep")
    custom_prompt: Optional[str] = Field(None, max_length=2000)


# ── Response models ─────────────────────────────────────────────

class CitationOut(BaseModel):
    """Citazione nell'output."""
    citation_id: str
    source_file: str
    page_number: int
    text_snippet: str
    score: float


class ChatResponse(BaseModel):
    """Risposta chat completa."""
    answer: str
    citations: List[CitationOut] = Field(default_factory=list)
    backend_used: str = ""
    model: str = ""
    latency_s: float = 0.0
    quality: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UploadResponse(BaseModel):
    """Conferma upload documento."""
    filename: str
    file_type: str
    pages: int
    chunks: int
    message: str


class DocumentInfo(BaseModel):
    """Informazioni documento indicizzato."""
    filename: str
    file_type: str
    chunks: int
    indexed_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Lista documenti."""
    documents: List[DocumentInfo]
    total: int


class VisionResponse(BaseModel):
    """Risposta analisi immagine."""
    analysis: str
    backend: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Stato di salute del sistema."""
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_s: float
    backends: Dict[str, bool] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ErrorResponse(BaseModel):
    """Errore leggibile per l'utente (nessun stack trace)."""
    error: str
    detail: str = ""
    suggestion: str = ""


# ── Chat Session models ─────────────────────────────────────────

class ChatSessionSummary(BaseModel):
    """Riepilogo sessione per lista."""
    session_id: str
    title: str
    modalita: str = "generale"
    message_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class ChatSessionDetail(BaseModel):
    """Sessione completa con messaggi."""
    session_id: str
    title: str
    modalita: str = "generale"
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class ChatSessionListResponse(BaseModel):
    """Lista sessioni."""
    sessions: List[ChatSessionSummary]
    total: int
