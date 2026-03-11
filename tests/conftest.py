"""Pytest fixtures condivise per tutti i test."""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def tmp_data_dir():
    """Directory temporanea per file di test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_text():
    """Testo di esempio in italiano per test di chunking/retrieval."""
    return (
        "Il Rinascimento fu un movimento culturale che ebbe origine in Italia "
        "nel XIV secolo. Si diffuse poi in tutta Europa nei secoli XV e XVI. "
        "L'arte rinascimentale si caratterizzava per il ritorno ai modelli classici "
        "dell'antica Grecia e Roma. Leonardo da Vinci, Michelangelo e Raffaello "
        "sono considerati i tre grandi maestri del Rinascimento italiano. "
        "La prospettiva lineare, sviluppata da Brunelleschi, rivoluzionò la "
        "rappresentazione dello spazio nelle arti visive. "
        "Il Rinascimento segnò anche una nuova concezione dell'uomo, posto al "
        "centro dell'universo secondo la filosofia umanistica."
    )


@pytest.fixture
def sample_chunks(sample_text):
    """Lista di chunk di esempio per test di retrieval."""
    sentences = sample_text.split(". ")
    from ingestion.chunker import Chunk
    return [
        Chunk(
            chunk_id=f"test_chunk_{i}",
            text=s.strip() + ".",
            source_file="test.pdf",
            page_number=1,
            chunk_index=i,
        )
        for i, s in enumerate(sentences)
        if s.strip()
    ]
