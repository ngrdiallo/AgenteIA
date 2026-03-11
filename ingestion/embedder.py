"""
Embedding service: wrapper per modelli multilingual sentence-transformers.
Usa il prefisso "query: " / "passage: " richiesto dai modelli E5.
"""

import logging
from typing import List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Servizio embedding con lazy loading del modello.
    Compatible con i modelli intfloat/multilingual-e5-* che richiedono prefissi.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension: Optional[int] = None
        # I modelli E5 richiedono prefissi specifici
        self._is_e5 = "e5" in self.model_name.lower()

    @property
    def model(self):
        """Lazy load del modello al primo utilizzo."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Caricamento modello embedding: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding pronto: {self._dimension} dimensioni")
        return self._model

    @property
    def dimension(self) -> int:
        """Dimensione del vettore embedding."""
        if self._dimension is None:
            _ = self.model  # Forza caricamento
        return self._dimension  # type: ignore

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Genera embedding per documenti (passage).
        Per modelli E5: aggiunge prefisso "passage: ".
        """
        if not texts:
            return np.array([])

        prefixed = [f"passage: {t}" for t in texts] if self._is_e5 else texts
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Genera embedding per una query di ricerca.
        Per modelli E5: aggiunge prefisso "query: ".
        """
        prefixed = f"query: {query}" if self._is_e5 else query
        embedding = self.model.encode(
            [prefixed],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Genera embedding per multiple query."""
        if not queries:
            return np.array([])
        prefixed = [f"query: {q}" for q in queries] if self._is_e5 else queries
        return self.model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
