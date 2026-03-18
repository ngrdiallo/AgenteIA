"""
Re-ranker: riordina i risultati del retrieval con un modello cross-encoder.
Supporta: cross-encoder locale (sentence-transformers) o Cohere Rerank API.
"""

import logging
from typing import List, Optional

from config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Re-ranking dei risultati di ricerca con cross-encoder.

    Provider configurabili:
    - "local": cross-encoder via sentence-transformers (offline, gratuito)
    - "cohere": Cohere Rerank API (cloud, richiede COHERE_API_KEY)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.provider = provider or settings.RERANKER_PROVIDER
        self.model_name = model_name or settings.RERANKER_MODEL
        self._cross_encoder = None
        self._cohere_client = None

    # Soglia minima: cross-encoder ms-marco restituisce score negativi per
    # risultati irrilevanti. Sotto questa soglia = nessuna corrispondenza reale.
    RERANK_MIN_SCORE = -2.0

    def rerank(
        self,
        query: str,
        results: list,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list:
        """
        Riordina i risultati di ricerca in base alla rilevanza.

        Args:
            query: testo della query utente
            results: lista di SearchResult
            top_k: quanti risultati ritornare dopo reranking
            min_score: soglia minima di rilevanza (default RERANK_MIN_SCORE)
        Returns:
            Lista di SearchResult riordinata per rilevanza (solo quelli sopra soglia)
        """
        if top_k is None:
            top_k = settings.RERANK_TOP_K

        if not results:
            return []

        if len(results) <= 1:
            return results[:top_k]

        try:
            _min = min_score if min_score is not None else self.RERANK_MIN_SCORE
            if self.provider == "cohere":
                return self._rerank_cohere(query, results, top_k)
            else:
                return self._rerank_local(query, results, top_k, _min)
        except Exception as e:
            logger.warning(f"Re-ranking fallito ({self.provider}): {e}. Ritorno ranking originale.")
            return results[:top_k]

    def _rerank_local(self, query: str, results: list, top_k: int, min_score: float = -2.0) -> list:
        """Re-ranking con cross-encoder locale."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Caricamento cross-encoder: {self.model_name}")
            self._cross_encoder = CrossEncoder(self.model_name)

        # Prepara coppie (query, documento)
        pairs = [(query, r.text) for r in results]
        scores = self._cross_encoder.predict(pairs)

        # Assegna score e riordina
        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Filtra per soglia minima di rilevanza
        reranked = []
        for result, score in scored[:top_k]:
            if float(score) < min_score:
                logger.debug(f"Scartato chunk {result.chunk_id}: score={score:.3f} < soglia={min_score}")
                continue
            from retrieval.hybrid_search import SearchResult
            reranked.append(SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                source_file=result.source_file,
                page_number=result.page_number,
                metadata={**result.metadata, "rerank_score": float(score)},
            ))

        logger.debug(f"Reranking locale: {len(results)} → {len(reranked)} risultati")
        return reranked

    def _rerank_cohere(self, query: str, results: list, top_k: int) -> list:
        """Re-ranking con Cohere Rerank API."""
        if not settings.COHERE_API_KEY:
            logger.warning("COHERE_API_KEY non configurata, fallback a reranker locale")
            return self._rerank_local(query, results, top_k)

        if self._cohere_client is None:
            import cohere
            self._cohere_client = cohere.Client(settings.COHERE_API_KEY)

        docs = [r.text for r in results]
        response = self._cohere_client.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=docs,
            top_n=top_k,
        )

        reranked = []
        for item in response.results:
            r = results[item.index]
            from retrieval.hybrid_search import SearchResult
            reranked.append(SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=float(item.relevance_score),
                source_file=r.source_file,
                page_number=r.page_number,
                metadata={**r.metadata, "rerank_score": float(item.relevance_score)},
            ))

        logger.debug(f"Reranking Cohere: {len(results)} → {len(reranked)} risultati")
        return reranked
