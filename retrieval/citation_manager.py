"""
Citation manager: genera citazioni precise con testo sorgente esatto.
PORTED FROM: AgenteIA-Production/src/rag_engine.py (CitationManager, migliorato)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Una singola citazione con tutti i dati per la verifica."""
    citation_id: str
    source_file: str
    page_number: int
    chunk_id: str
    text_snippet: str           # Testo sorgente esatto (non troncato)
    score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CitationManager:
    """
    Gestisce citazioni verificabili per ogni risposta.
    Migliora il sistema precedente aggiungendo:
    - Testo sorgente completo (non troncato a 100 char)
    - Chunk ID per tracciabilità
    - Score di rilevanza
    - Formato inline + footnote
    """

    def __init__(self):
        self._citations: List[Citation] = []

    def clear(self):
        """Reset per nuova risposta."""
        self._citations = []

    def build_citations(self, search_results: list) -> List[Citation]:
        """
        Costruisce citazioni dai risultati di ricerca.
        Ogni risultato diventa una citazione verificabile.
        """
        self.clear()

        for i, r in enumerate(search_results):
            citation = Citation(
                citation_id=f"cite_{i + 1}",
                source_file=r.source_file,
                page_number=r.page_number,
                chunk_id=r.chunk_id,
                text_snippet=r.text,  # Testo completo, non troncato
                score=r.score,
            )
            self._citations.append(citation)

        return self._citations

    @property
    def citations(self) -> List[Citation]:
        return self._citations

    def format_inline(self, citation: Citation) -> str:
        """Formatta una citazione per uso inline nel testo."""
        return f"[📄 {citation.source_file} | p. {citation.page_number}]"

    def format_all_inline(self) -> List[str]:
        """Ritorna lista di marker inline per tutte le citazioni."""
        return [self.format_inline(c) for c in self._citations]

    def format_footnotes(self) -> str:
        """
        Genera sezione footnote con tutte le citazioni.
        Include testo sorgente per verifica.
        """
        if not self._citations:
            return ""

        lines = ["\n---", "**📚 Fonti Citate**\n"]
        for c in self._citations:
            snippet = c.text_snippet[:300]
            if len(c.text_snippet) > 300:
                snippet += "…"
            lines.append(
                f"**[{c.citation_id}]** {c.source_file} (p. {c.page_number})\n"
                f"> \"{snippet}\"\n"
            )

        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict]:
        """Serializza citazioni per la risposta API."""
        return [
            {
                "id": c.citation_id,
                "source_file": c.source_file,
                "page_number": c.page_number,
                "chunk_id": c.chunk_id,
                "text_snippet": c.text_snippet[:500],
                "score": round(c.score, 4),
            }
            for c in self._citations
        ]
