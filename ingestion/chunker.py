"""
Chunking semantico: divide documenti in chunk rispettando confini frasali.
Migliora retrieval rispetto al chunking puramente token-based del sistema precedente.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

# Scarica dati NLTK al primo utilizzo
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    try:
        import nltk
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        import nltk
        nltk.download("punkt_tab", quiet=True)
    _nltk_ready = True


@dataclass
class Chunk:
    """Un singolo chunk di testo con metadata completi per citazione."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """
    Chunking semantico basato su confini frasali.

    Strategia:
    1. Tokenizza il testo in frasi (NLTK, italiano)
    2. Aggrega frasi fino al raggiungimento di chunk_size token
    3. Aggiunge overlap di chunk_overlap frasi dal chunk precedente
    4. Preserva tabelle come chunk atomici (non vengono spezzate)
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list) -> List[Chunk]:
        """
        Riceve lista di DocumentPage e ritorna lista di Chunk.
        Ogni pagina viene processata separatamente per preservare il page_number.
        """
        from ingestion.loaders import DocumentPage

        all_chunks: List[Chunk] = []
        chunk_index = 0

        for page in pages:
            if not isinstance(page, DocumentPage):
                continue

            # Chunk per tabelle (atomici, non spezzati)
            if page.tables:
                for table in page.tables:
                    table_text = self._table_to_text(table)
                    if table_text.strip():
                        c = Chunk(
                            chunk_id=self._make_id(page.source_file, page.page_number, chunk_index),
                            text=table_text,
                            source_file=page.source_file,
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            metadata={"type": "table", "total_pages": page.total_pages},
                        )
                        all_chunks.append(c)
                        chunk_index += 1

            # Chunk per testo
            if page.text.strip():
                text_chunks = self._chunk_text(page.text)
                for text in text_chunks:
                    c = Chunk(
                        chunk_id=self._make_id(page.source_file, page.page_number, chunk_index),
                        text=text,
                        source_file=page.source_file,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={"type": "text", "total_pages": page.total_pages},
                    )
                    all_chunks.append(c)
                    chunk_index += 1

        logger.info(f"Chunking completato: {len(all_chunks)} chunk da {len(pages)} pagine")
        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Divide testo in chunk basati su frasi con overlap."""
        _ensure_nltk()
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text, language="italian")
        if not sentences:
            return [text] if text.strip() else []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._count_tokens(sent)

            # Se una singola frase supera il chunk_size, la includi comunque
            if sent_tokens > self.chunk_size and not current_sentences:
                chunks.append(sent.strip())
                continue

            # Se aggiungere questa frase supera il budget
            if current_tokens + sent_tokens > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences).strip())

                # Overlap: mantieni le ultime N frasi
                overlap_count = min(self.chunk_overlap, len(current_sentences))
                if overlap_count > 0:
                    current_sentences = current_sentences[-overlap_count:]
                    current_tokens = sum(self._count_tokens(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sent)
            current_tokens += sent_tokens

        # Ultimo chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences).strip())

        return [c for c in chunks if c.strip()]

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Stima veloce del numero di token (circa 1 token = 4 char per italiano)."""
        return max(1, len(text) // 4)

    @staticmethod
    def _table_to_text(table: list) -> str:
        """Converte tabella (lista di righe) in testo leggibile."""
        if not table:
            return ""
        lines = []
        for row in table:
            if isinstance(row, (list, tuple)):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)

    @staticmethod
    def _make_id(source_file: str, page: int, index: int) -> str:
        """Genera ID univoco per chunk."""
        raw = f"{source_file}::p{page}::c{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
