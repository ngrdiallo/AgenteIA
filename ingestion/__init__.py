"""Ingestion pipeline: caricamento documenti → chunking → embedding → indexing."""

from ingestion.loaders import DocumentLoader, DocumentPage
from ingestion.chunker import SemanticChunker, Chunk
from ingestion.embedder import EmbeddingService
from ingestion.indexer import DocumentIndexer

__all__ = [
    "DocumentLoader", "DocumentPage",
    "SemanticChunker", "Chunk",
    "EmbeddingService",
    "DocumentIndexer",
]
