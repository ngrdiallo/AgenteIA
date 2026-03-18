"""
Indexer: gestione ChromaDB per storage vettoriale persistente.
Mantiene anche i dati necessari per ricostruire l'indice BM25.
"""

import logging
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """
    Gestisce l'indice vettoriale ChromaDB persistente.
    Fornisce anche accesso ai testi raw per BM25.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "documents",
    ):
        import chromadb

        self.persist_dir = persist_dir or str(settings.STORAGE_DIR / "chroma")
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB inizializzato: {self._collection.count()} documenti in '{self.collection_name}'"
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(
        self,
        chunk_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
    ) -> int:
        """
        Aggiunge chunk all'indice. Salta duplicati (stesso chunk_id).
        Ritorna il numero di chunk effettivamente aggiunti.
        """
        if not chunk_ids:
            return 0

        # ChromaDB gestisce upsert nativamente
        # Lavora in batch da 500 per evitare limiti di memoria
        batch_size = 500
        added = 0
        for i in range(0, len(chunk_ids), batch_size):
            end = min(i + batch_size, len(chunk_ids))
            self._collection.upsert(
                ids=chunk_ids[i:end],
                documents=texts[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
            added += end - i

        logger.info(f"Indicizzati {added} chunk (totale: {self._collection.count()})")
        # Invalidate doc info cache
        if hasattr(self, '_doc_info_cache'):
            self._doc_info_cache.clear()
        return added

    def delete_by_source(self, source_file: str) -> int:
        """Rimuove tutti i chunk di un documento sorgente."""
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=[],
            )
            ids_to_delete = results["ids"]
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info(f"Rimossi {len(ids_to_delete)} chunk per '{source_file}'")
                # Invalidate doc info cache
                if hasattr(self, '_doc_info_cache'):
                    self._doc_info_cache.clear()
                return len(ids_to_delete)
        except Exception as e:
            logger.error(f"Errore eliminazione chunk per '{source_file}': {e}")
        return 0

    def query_dense(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> Dict:
        """
        Ricerca dense (cosine similarity) su ChromaDB.
        Ritorna dict con ids, documents, metadatas, distances.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_all_documents(self) -> Dict:
        """
        Ritorna tutti i documenti per ricostruire indice BM25.
        Returns dict con ids, documents, metadatas.
        """
        if self._collection.count() == 0:
            return {"ids": [], "documents": [], "metadatas": []}

        return self._collection.get(
            include=["documents", "metadatas"],
        )

    def list_source_files(self) -> List[str]:
        """Lista dei file sorgente univoci nell'indice."""
        if self._collection.count() == 0:
            return []
        try:
            all_docs = self._collection.get(include=["metadatas"])
            files = set()
            for meta in all_docs["metadatas"]:
                if meta and "source_file" in meta:
                    files.add(meta["source_file"])
            return sorted(files)
        except Exception:
            return []

    def get_chunks_by_source(self, source_file: str) -> List[Dict]:
        """
        Ritorna TUTTI i chunk di un documento specifico, ordinati per pagina e chunk_index.
        Utile per analisi comprensive dove serve copertura completa.

        Returns:
            Lista di dict con keys: id, text, metadata (page_number, chunk_index, source_file)
        """
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=["documents", "metadatas"],
            )
            if not results["ids"]:
                return []

            chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                chunks.append({
                    "id": chunk_id,
                    "text": results["documents"][i] if results["documents"] else "",
                    "metadata": meta,
                    "page_number": meta.get("page_number", 0),
                    "chunk_index": meta.get("chunk_index", i),
                })

            # Ordina per pagina e poi per chunk_index
            chunks.sort(key=lambda c: (c["page_number"], c["chunk_index"]))
            return chunks
        except Exception as e:
            logger.error(f"Errore get_chunks_by_source per '{source_file}': {e}")
            return []

    def get_document_info(self, source_file: str) -> Dict:
        """Info su un documento specifico — usa cache per evitare query ripetute."""
        # Cache check
        if not hasattr(self, '_doc_info_cache'):
            self._doc_info_cache = {}
        if source_file in self._doc_info_cache:
            return self._doc_info_cache[source_file]

        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=["metadatas"],
            )
            if not results["ids"]:
                return {}
            chunk_count = len(results["ids"])
            pages = set()
            for meta in results["metadatas"]:
                if meta and "page_number" in meta:
                    pages.add(meta["page_number"])
            info = {
                "source_file": source_file,
                "chunk_count": chunk_count,
                "pages": sorted(pages),
                "total_pages": max(pages) if pages else 0,
            }
            self._doc_info_cache[source_file] = info
            return info
        except Exception:
            return {}

    def clear(self):
        """Svuota l'intera collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Invalidate doc info cache
        if hasattr(self, '_doc_info_cache'):
            self._doc_info_cache.clear()
        logger.info("Indice svuotato")
