"""Test completo IAGestioneArte — verifica import, inizializzazione e logica."""
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append((name, traceback.format_exc()))
        failed += 1


# ══════════════════════════════════════════════════════════
# TEST 1: Config
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 1: Config")
print("=" * 60)


def test_config_load():
    from config import settings
    assert settings.APP_NAME == "IAGestioneArte"
    assert settings.VERSION == "1.0.0"
    assert settings.CHUNK_SIZE > 0
    assert settings.TOP_K > 0


def test_config_paths():
    from config import settings
    assert settings.PROJECT_ROOT.exists()
    assert settings.STORAGE_DIR is not None
    assert settings.LOGS_DIR is not None


def test_prompts():
    from config.prompts import get_system_prompt, MODALITA_LIST
    assert len(MODALITA_LIST) == 8
    ids = [m["id"] for m in MODALITA_LIST]
    assert "generale" in ids
    assert "storico_artistico" in ids
    for m in MODALITA_LIST:
        prompt = get_system_prompt(m["id"])
        assert len(prompt) > 100, f"Prompt '{m['id']}' troppo corto"


test("Config load", test_config_load)
test("Config paths", test_config_paths)
test("Prompts (8 modalita)", test_prompts)


# ══════════════════════════════════════════════════════════
# TEST 2: Ingestion
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 2: Ingestion")
print("=" * 60)


def test_loader_import():
    from ingestion.loaders import DocumentLoader, DocumentPage
    loader = DocumentLoader()
    assert loader is not None


def test_chunker_basic():
    from ingestion.chunker import SemanticChunker, Chunk
    from ingestion.loaders import DocumentPage
    chunker = SemanticChunker()
    pages = [DocumentPage(
        text="Questo è un testo di prova per il chunker semantico. "
             "Contiene alcune frasi di esempio. Il Rinascimento italiano "
             "ha prodotto capolavori senza precedenti nell'arte mondiale.",
        page_number=1,
        source_file="test.pdf",
        file_type="pdf",
        total_pages=1,
    )]
    chunks = chunker.chunk_pages(pages)
    assert len(chunks) >= 1
    assert chunks[0].source_file == "test.pdf"
    assert chunks[0].chunk_id != ""
    assert len(chunks[0].text) > 10


def test_embedder():
    from ingestion.embedder import EmbeddingService
    embedder = EmbeddingService()
    # Test query embedding
    vec = embedder.embed_query("Test query per l'arte rinascimentale")
    assert len(vec) > 0  # Should be 768 for E5-base
    # Test document embedding
    docs = embedder.embed_documents(["Documento di prova sulla storia dell'arte."])
    assert docs.shape[0] == 1
    assert docs.shape[1] > 0


def test_indexer():
    from ingestion.indexer import DocumentIndexer
    indexer = DocumentIndexer(collection_name="test_collection")
    # Cleanup
    indexer.clear()
    assert indexer.count == 0
    # Add
    from ingestion.embedder import EmbeddingService
    embedder = EmbeddingService()
    texts = ["Caravaggio e il chiaroscuro.", "Bernini e la scultura barocca."]
    embeddings = embedder.embed_documents(texts).tolist()
    indexer.add_chunks(
        chunk_ids=["c1", "c2"],
        texts=texts,
        embeddings=embeddings,
        metadatas=[
            {"source_file": "test.pdf", "page_number": 1, "chunk_index": 0},
            {"source_file": "test.pdf", "page_number": 1, "chunk_index": 1},
        ],
    )
    assert indexer.count == 2
    # Query
    qvec = embedder.embed_query("chiaroscuro")
    results = indexer.query_dense(qvec, top_k=2)
    assert len(results["ids"][0]) == 2
    # List files
    files = indexer.list_source_files()
    assert "test.pdf" in files
    # Cleanup
    indexer.clear()
    assert indexer.count == 0


test("Loader import", test_loader_import)
test("Chunker semantic", test_chunker_basic)
test("Embedder E5 (query+doc)", test_embedder)
test("Indexer ChromaDB CRUD", test_indexer)


# ══════════════════════════════════════════════════════════
# TEST 3: Retrieval
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 3: Retrieval")
print("=" * 60)


def test_hybrid_search():
    from ingestion.embedder import EmbeddingService
    from ingestion.indexer import DocumentIndexer
    from retrieval.hybrid_search import HybridSearchEngine, SearchResult

    embedder = EmbeddingService()
    indexer = DocumentIndexer(collection_name="test_search")
    indexer.clear()

    # Popola
    texts = [
        "Leonardo da Vinci dipinse la Gioconda nel XVI secolo.",
        "Michelangelo scolpì il David a Firenze.",
        "Raffaello decorò le Stanze Vaticane con affreschi.",
    ]
    embs = embedder.embed_documents(texts).tolist()
    indexer.add_chunks(
        chunk_ids=["s1", "s2", "s3"],
        texts=texts,
        embeddings=embs,
        metadatas=[
            {"source_file": "arte.pdf", "page_number": 1, "chunk_index": i}
            for i in range(3)
        ],
    )

    engine = HybridSearchEngine(indexer=indexer, embedder=embedder)
    engine.rebuild_bm25_index()
    results = engine.search("Gioconda Leonardo", top_k=3)
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    # Il primo risultato dovrebbe contenere "Leonardo" o "Gioconda"
    assert "Leonardo" in results[0].text or "Gioconda" in results[0].text

    indexer.clear()


def test_reranker():
    from retrieval.reranker import Reranker
    from retrieval.hybrid_search import SearchResult
    reranker = Reranker()
    results = [
        SearchResult(chunk_id="r1", text="L'arte barocca in Italia.", score=0.8,
                     source_file="art.pdf", page_number=1),
        SearchResult(chunk_id="r2", text="La cucina italiana moderna.", score=0.7,
                     source_file="food.pdf", page_number=1),
    ]
    reranked = reranker.rerank("arte barocca", results, top_k=2)
    assert len(reranked) > 0
    # Il reranker dovrebbe favorire il testo sull'arte
    assert "arte" in reranked[0].text.lower() or "barocca" in reranked[0].text.lower()


def test_context_builder():
    from retrieval.context_builder import ContextBuilder
    from retrieval.hybrid_search import SearchResult
    builder = ContextBuilder()
    results = [
        SearchResult(chunk_id="cb1", text="Il Colosseo è un anfiteatro romano.",
                     score=0.9, source_file="roma.pdf", page_number=5),
    ]
    prompt = builder.build(query="Parlami del Colosseo", search_results=results)
    assert "Colosseo" in prompt
    assert "anfiteatro" in prompt


def test_citation_manager():
    from retrieval.citation_manager import CitationManager
    from retrieval.hybrid_search import SearchResult
    cm = CitationManager()
    results = [
        SearchResult(chunk_id="cit1", text="Giotto affrescò la Cappella degli Scrovegni.",
                     score=0.95, source_file="giotto.pdf", page_number=12),
    ]
    citations = cm.build_citations(results)
    assert len(citations) == 1
    assert citations[0].source_file == "giotto.pdf"
    assert citations[0].page_number == 12
    inline = cm.format_inline(citations[0])
    assert "giotto.pdf" in inline


test("Hybrid search (dense+BM25+RRF)", test_hybrid_search)
test("Reranker cross-encoder", test_reranker)
test("Context builder", test_context_builder)
test("Citation manager", test_citation_manager)


# ══════════════════════════════════════════════════════════
# TEST 4: LLM
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 4: LLM Layer")
print("=" * 60)


def test_orchestrator_init():
    from llm.orchestrator import LLMOrchestrator, LLMResponse
    llm = LLMOrchestrator()
    assert llm.BACKEND_CHAIN == [
        "openrouter", "groq", "mistral", "gemini", "huggingface", "deepseek", "ollama"
    ]
    # Complete senza API key deve ritornare messaggio di errore gracefully
    resp = llm.complete("Test", "Rispondi in italiano.")
    assert isinstance(resp, LLMResponse)
    assert resp.backend_used in ("none", *llm.BACKEND_CHAIN)


def test_quality_evaluator():
    from llm.quality_evaluator import ResponseQualityEvaluator
    ev = ResponseQualityEvaluator()
    # Risposta corta semplice
    score = ev.evaluate("Sì.", "Cos'è il Rinascimento?")
    assert score.needs_escalation is True
    assert score.word_count == 1
    # Risposta lunga e dettagliata
    long_resp = (
        "Il Rinascimento è un movimento culturale nato in Italia nel XIV secolo. "
        "Storicamente ha rappresentato una svolta fondamentale. "
        "Artisticamente, confrontando il periodo medievale con quello rinascimentale, "
        "si nota un'influenza profonda della pittura fiamminga. "
        "Analizzando la tecnica della prospettiva lineare sviluppata da Brunelleschi, "
        "si comprende come il contesto fiorentino abbia favorito l'innovazione. "
        "L'iconografia cristiana fu reinterpretata con nuovi significati. "
        "Quindi questo periodo segna il passaggio verso la modernità. "
    ) * 3
    score2 = ev.evaluate(long_resp, "Confronta il Rinascimento con il Medioevo")
    assert score2.depth_score > 0.4
    assert score2.word_count > 100


def test_italian_filter():
    from llm.italian_filter import ItalianFilter
    filt = ItalianFilter(enabled=True)
    # Testo tutto italiano: nessuna modifica
    ita = "Il Rinascimento italiano è un periodo di grande splendore artistico."
    assert filt.filter(ita) == ita
    # Testo con "Hello" → "Ciao"
    mixed = "Hello, benvenuto nel museo. La collezione è magnifica."
    result = filt.filter(mixed)
    assert "Ciao" in result or "benvenuto" in result


def test_vision_init():
    from llm.vision import VisionAnalyzer
    v = VisionAnalyzer()
    assert "quick" in v.ANALYSIS_PROMPTS
    assert "standard" in v.ANALYSIS_PROMPTS
    assert "deep" in v.ANALYSIS_PROMPTS


test("LLM Orchestrator init + fallback", test_orchestrator_init)
test("Quality evaluator", test_quality_evaluator)
test("Italian filter", test_italian_filter)
test("Vision analyzer init", test_vision_init)


# ══════════════════════════════════════════════════════════
# TEST 5: API
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 5: API Layer")
print("=" * 60)


def test_pydantic_models():
    from api.models import ChatRequest, ChatResponse, UploadResponse, HealthResponse
    req = ChatRequest(query="Test domanda", modalita="generale")
    assert req.use_rag is True
    resp = ChatResponse(answer="Risposta test", backend_used="groq")
    assert resp.latency_s == 0.0


def test_app_creation():
    from api.routes import create_app
    app = create_app()
    assert app is not None
    # Verifica che le route esistano
    routes = [r.path for r in app.routes]
    assert "/api/chat" in routes
    assert "/api/documents/upload" in routes
    assert "/api/documents" in routes
    assert "/api/health" in routes
    assert "/api/modalita" in routes


test("Pydantic models", test_pydantic_models)
test("FastAPI app creation + routes", test_app_creation)


# ══════════════════════════════════════════════════════════
# TEST 6: System
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TEST GRUPPO 6: System Utilities")
print("=" * 60)


def test_cleanup_manager():
    from system.cleanup import CleanupManager
    cm = CleanupManager()
    stats = cm.run_once()
    assert "temp_deleted" in stats
    assert "logs_deleted" in stats


def test_health_monitor():
    from system.health import HealthMonitor
    hm = HealthMonitor()
    status = hm.check()
    assert status.status in ("healthy", "degraded", "unhealthy")
    assert status.disk_total_gb > 0
    assert "disk_ok" in status.checks


def test_logging():
    from system.logging_config import setup_logging
    # Non dovrebbe crashare
    setup_logging()


test("Cleanup manager", test_cleanup_manager)
test("Health monitor", test_health_monitor)
test("Logging setup", test_logging)


# ══════════════════════════════════════════════════════════
# RIEPILOGO
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RIEPILOGO: {passed} PASSED, {failed} FAILED su {passed + failed} test")
print("=" * 60)

if errors:
    print("\nDETTAGLI ERRORI:")
    for name, tb in errors:
        print(f"\n--- {name} ---")
        print(tb)

sys.exit(0 if failed == 0 else 1)
