"""
Hybrid search: combina dense retrieval (ChromaDB) + sparse retrieval (BM25).
Fusione via Reciprocal Rank Fusion (RRF) per risultati migliori.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Risultato di ricerca con score fuso."""
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridSearchEngine:
    """
    Ricerca ibrida Dense + BM25 con fusione RRF.

    Dense: cosine similarity via ChromaDB
    Sparse: BM25 (rank_bm25) con tokenizzazione semplice
    Fusion: Reciprocal Rank Fusion — RRF(d) = Σ 1/(k + rank_i(d))
    """

    RRF_K = 60  # Costante standard per RRF

    def __init__(self, indexer, embedder):
        """
        Args:
            indexer: DocumentIndexer (ChromaDB)
            embedder: EmbeddingService
        """
        self.indexer = indexer
        self.embedder = embedder
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_texts: List[str] = []
        self._bm25_metas: List[Dict] = []

    def rebuild_bm25_index(self):
        """
        Ricostruisce l'indice BM25 in memoria partendo da ChromaDB.
        Da chiamare dopo ogni aggiunta/rimozione di documenti.
        """
        from rank_bm25 import BM25Okapi

        all_docs = self.indexer.get_all_documents()
        ids = all_docs.get("ids", [])
        texts = all_docs.get("documents", [])
        metas = all_docs.get("metadatas", [])

        if not texts:
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_texts = []
            self._bm25_metas = []
            logger.info("BM25 index vuoto (nessun documento)")
            return

        # Tokenizzazione semplice per BM25 (lowercase + split)
        tokenized = [self._tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_ids = ids
        self._bm25_texts = texts
        self._bm25_metas = metas

        logger.info(f"BM25 index ricostruito: {len(ids)} documenti")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[SearchResult]:
        """
        Ricerca ibrida: Dense + BM25 → RRF fusion.

        Args:
            query: testo della query
            top_k: numero risultati da ritornare
            dense_weight: peso del ranking dense nella fusione
            sparse_weight: peso del ranking BM25 nella fusione
        """
        if top_k is None:
            top_k = settings.TOP_K

        # Se non ci sono documenti, ritorna vuoto
        if self.indexer.count == 0:
            return []

        # Preprocessa la query: usa sia la query originale che una versione
        # con i soli termini chiave (rimuovendo frasi meta come "cosa dice il documento su")
        clean_query = self._extract_key_terms(query)

        # --- Dense search (ChromaDB) con query originale ---
        query_emb = self.embedder.embed_query(query)
        dense_results = self.indexer.query_dense(
            query_embedding=query_emb.tolist(),
            top_k=top_k * 2,
        )
        dense_ranking = self._parse_chroma_results(dense_results)

        # --- Dense search aggiuntiva con termini chiave (se diversa) ---
        if clean_query and clean_query.lower() != query.lower():
            key_emb = self.embedder.embed_query(clean_query)
            key_results = self.indexer.query_dense(
                query_embedding=key_emb.tolist(),
                top_k=top_k,
            )
            key_parsed = self._parse_chroma_results(key_results)
            # Merge: aggiungi risultati non già presenti
            seen_ids = {r.chunk_id for r in dense_ranking}
            for r in key_parsed:
                if r.chunk_id not in seen_ids:
                    dense_ranking.append(r)
                    seen_ids.add(r.chunk_id)
            logger.debug(f"Key-terms search aggiuntiva '{clean_query}': +{len(key_parsed)} candidati")

        # --- Sparse search (BM25) ---
        sparse_ranking = self._search_bm25(query, top_k=top_k * 2)

        # --- Pre-filter per score threshold sui ranking originali ---
        threshold = settings.SCORE_THRESHOLD
        dense_filtered = [r for r in dense_ranking if r.score >= threshold]
        sparse_filtered = sparse_ranking  # BM25 scores non sono normalizzati 0-1, non filtrare

        # --- RRF Fusion ---
        if dense_filtered and sparse_filtered:
            fused = self._rrf_fusion(
                dense_filtered, sparse_filtered,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
        elif dense_filtered:
            fused = dense_filtered
        elif sparse_filtered:
            fused = sparse_filtered
        elif dense_ranking:
            # Fallback: se il threshold era troppo alto, usa i ranking originali
            fused = self._rrf_fusion(
                dense_ranking, sparse_ranking,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            ) if sparse_ranking else dense_ranking
        elif sparse_ranking:
            fused = sparse_ranking
        else:
            return []

        results = fused[:top_k]

        logger.debug(
            f"Hybrid search: {len(dense_ranking)} dense + {len(sparse_ranking)} sparse "
            f"→ {len(results)} fused (threshold={threshold})"
        )
        return results

    def _search_bm25(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Ricerca BM25 sparse."""
        if self._bm25 is None or not self._bm25_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Prendi top-k indici
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metas[idx] if idx < len(self._bm25_metas) else {}
            results.append(SearchResult(
                chunk_id=self._bm25_ids[idx],
                text=self._bm25_texts[idx],
                score=float(scores[idx]),
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                metadata=meta,
            ))
        return results

    def _parse_chroma_results(self, results: Dict) -> List[SearchResult]:
        """Converte risultati ChromaDB in SearchResult."""
        parsed = []
        if not results or not results.get("ids"):
            return parsed

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            # ChromaDB ritorna distanze (minore = più simile), convertiamo in score
            score = max(0.0, 1.0 - distances[i])
            meta = metas[i] or {}
            parsed.append(SearchResult(
                chunk_id=ids[i],
                text=docs[i],
                score=score,
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 0),
                metadata=meta,
            ))
        return parsed

    def _rrf_fusion(
        self,
        ranking_a: List[SearchResult],
        ranking_b: List[SearchResult],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion: combina due ranking.
        RRF_score(d) = w_a / (k + rank_a(d)) + w_b / (k + rank_b(d))
        """
        scores: Dict[str, float] = {}
        result_map: Dict[str, SearchResult] = {}

        for rank, r in enumerate(ranking_a):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + dense_weight / (self.RRF_K + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(ranking_b):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0) + sparse_weight / (self.RRF_K + rank + 1)
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r

        # Ordina per score RRF decrescente
        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        fused = []
        for cid in sorted_ids:
            r = result_map[cid]
            fused.append(SearchResult(
                chunk_id=r.chunk_id,
                text=r.text,
                score=scores[cid],
                source_file=r.source_file,
                page_number=r.page_number,
                metadata=r.metadata,
            ))
        return fused

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenizzazione semplice per BM25: lowercase + split su whitespace."""
        return text.lower().split()

    @staticmethod
    def _extract_key_terms(query: str) -> str:
        """
        Rimuove frasi meta/filler dalla query per estrarre i termini chiave.
        Es: "Cosa dice il documento su Brunelleschi?" → "Brunelleschi"
            "Di cosa parla il file riguardo alla prospettiva?" → "prospettiva"
            "Brunelleschi" → "Brunelleschi" (invariato)
        """
        import re
        text = query.strip()
        # Rimuovi frasi meta italiane comuni
        meta_patterns = [
            r"^(?:cosa|che cosa)\s+(?:dice|racconta|spiega|descrive|parla)\s+(?:il|la|lo|i|le|gli)?\s*(?:documento|file|pdf|testo|libro)\s*(?:su|riguardo|circa|in merito a|a proposito di)?\s*",
            r"^(?:di cosa|di che)\s+(?:parla|tratta)\s+(?:il|la|lo)?\s*(?:documento|file|pdf|testo)?\s*(?:riguardo|circa|su|in merito a|a proposito di)?\s*",
            r"^(?:parlami|dimmi|spiegami|raccontami)\s+(?:di|del|della|dello|dei|delle|degli)?\s*",
            r"^(?:che|cosa|quali)\s+(?:informazioni|dettagli|dati)\s+(?:ci sono|contiene|ha)\s+(?:su|riguardo|circa)?\s*",
            r"^(?:puoi|potresti)\s+(?:dirmi|spiegarmi|descrivere)\s+(?:cosa|che cosa)?\s*(?:dice|parla)?\s*(?:su|riguardo)?\s*",
        ]
        for pattern in meta_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        # Rimuovi punteggiatura finale
        text = text.rstrip("?!.,;:")
        return text.strip() if text else query.strip()
