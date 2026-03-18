"""
API routes FastAPI con SSE streaming, upload, CRUD documenti, health.

Inizializza tutti i servizi al startup tramite lifespan context manager.
"""

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Token massimi per contesti comprehensive — bilanciamento qualità/compatibilità
MAX_COMPREHENSIVE_CONTEXT_TOKENS = 28_000

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import setup_middleware
from api.models import (
    ChatRequest,
    ChatResponse,
    ChatSessionDetail,
    ChatSessionListResponse,
    ChatSessionSummary,
    CitationOut,
    DocumentInfo,
    DocumentListResponse,
    ErrorResponse,
    HealthResponse,
    UploadResponse,
    VisionRequest,
    VisionResponse,
)
from config import settings
from config.prompts import get_system_prompt

logger = logging.getLogger(__name__)

# ── Startup / Shutdown ──────────────────────────────────────────

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inizializza servizi all'avvio, cleanup allo shutdown."""
    global _start_time
    _start_time = time.time()

    logger.info(f"🚀 Avvio {settings.APP_NAME} v{settings.VERSION}")

    # Lazy imports per non bloccare se dipendenze mancano
    from ingestion.embedder import EmbeddingService
    from ingestion.indexer import DocumentIndexer
    from ingestion.loaders import DocumentLoader
    from ingestion.chunker import SemanticChunker
    from retrieval.hybrid_search import HybridSearchEngine
    from retrieval.reranker import Reranker
    from retrieval.context_builder import ContextBuilder
    from retrieval.citation_manager import CitationManager
    from llm.orchestrator import LLMOrchestrator
    from llm.quality_evaluator import ResponseQualityEvaluator
    from llm.italian_filter import ItalianFilter
    from llm.vision import VisionAnalyzer
    from storage.chat_store import ChatStore

    # Inizializza servizi
    embedder = EmbeddingService()
    indexer = DocumentIndexer()
    loader = DocumentLoader()
    chunker = SemanticChunker()
    search_engine = HybridSearchEngine(indexer=indexer, embedder=embedder)
    reranker = Reranker()
    context_builder = ContextBuilder()
    citation_manager = CitationManager()
    llm = LLMOrchestrator()
    quality_eval = ResponseQualityEvaluator()
    italian_filter = ItalianFilter(enabled=settings.ITALIAN_ONLY)
    vision = VisionAnalyzer() if settings.VISION_ENABLED else None

    # Ricostruisci BM25 da documenti esistenti
    search_engine.rebuild_bm25_index()

    # Salva in app.state per accesso dalle routes
    app.state.embedder = embedder
    app.state.indexer = indexer
    app.state.loader = loader
    app.state.chunker = chunker
    app.state.search_engine = search_engine
    app.state.reranker = reranker
    app.state.context_builder = context_builder
    app.state.citation_manager = citation_manager
    app.state.llm = llm
    app.state.quality_eval = quality_eval
    app.state.italian_filter = italian_filter
    app.state.vision = vision

    # Chat store persistente
    chat_store = ChatStore()
    app.state.chat_store = chat_store

    logger.info("✅ Tutti i servizi inizializzati")

    # Warmup asincrono — evita cold-start sul primo request
    async def _warmup_providers():
        # Warmup speed chain (groq/cerebras) - 3s dopo boot
        await asyncio.sleep(3)
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: app.state.llm.complete(
                    "Rispondi solo OK",
                    routing_hint="speed"
                )
            )
            provider = result.backend_used if result and result.backend_used else None
            # Retry se rate-limited
            if not provider or provider == "none":
                await asyncio.sleep(5)
                result = await loop.run_in_executor(
                    None,
                    lambda: app.state.llm.complete(
                        "Rispondi solo OK",
                        routing_hint="speed"
                    )
                )
                provider = result.backend_used if result and result.backend_used else "no_provider"
            logger.info(f"🔥 Warmup speed: {provider}")
        except Exception as e:
            logger.warning(f"🔥 Warmup speed fallito: {e}")
        
        # Warmup large_context chain (openrouter) - 8s dopo boot
        await asyncio.sleep(5)
        try:
            loop = asyncio.get_event_loop()
            # Forza groq in degraded temporaneo per testare openrouter
            pool = app.state.llm.pool
            if "groq" in pool._states:
                import time
                pool._states["groq"].last_failure_time = time.time() - 20  # Already in cooldown
                pool._states["groq"].consecutive_failures = 2
            
            result = await loop.run_in_executor(
                None,
                lambda: app.state.llm.complete(
                    "Descrivi brevemente il Rinascimento italiano in 3 frasi.",
                    routing_hint="large_context"
                )
            )
            provider = result.backend_used if result and result.backend_used else None
            # Retry se rate-limited
            if not provider or provider == "none":
                await asyncio.sleep(5)
                result = await loop.run_in_executor(
                    None,
                    lambda: app.state.llm.complete(
                        "Descrivi brevemente il Rinascimento italiano in 3 frasi.",
                        routing_hint="large_context"
                    )
                )
                provider = result.backend_used if result and result.backend_used else "no_provider"
            logger.info(f"🔥 Warmup large_context: {provider}")
        except Exception as e:
            logger.warning(f"🔥 Warmup large_context fallito: {e}")

    asyncio.create_task(_warmup_providers())

    # Health probe passivo — ogni 5 minuti resetta provider degraded che non falliscono da 2+ min
    async def _health_probe_loop():
        import time
        from llm.backend_pool import CircuitState
        while True:
            await asyncio.sleep(300)  # 5 minuti
            try:
                pool = app.state.llm.pool
                for name, state in pool._states.items():
                    if state.state == CircuitState.DEGRADED:
                        elapsed = time.time() - (state.last_failure_time or 0)
                        if elapsed > 120:  # 2+ minuti senza fallimenti
                            state.consecutive_failures = max(0, state.consecutive_failures - 1)
                            if state.consecutive_failures < 3:
                                state.state = CircuitState.HEALTHY
                                logger.info(f"🔄 Health probe: {name} reset to healthy")
            except Exception as e:
                logger.warning(f"Health probe error: {e}")

    asyncio.create_task(_health_probe_loop())

    yield

    # Cleanup
    logger.info("Shutdown in corso...")


def create_app() -> FastAPI:
    """Factory dell'applicazione FastAPI."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    setup_middleware(app)
    _register_routes(app)
    _mount_static(app)
    return app


# ── Route registration ──────────────────────────────────────────

def _register_routes(app: FastAPI) -> None:

    # ── Chat (SSE streaming) ────────────────────────────────────

    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        """Chat con RAG + LLM. Ritorna SSE stream."""
        return StreamingResponse(
            _chat_stream(app, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat/sync", response_model=ChatResponse)
    async def chat_sync(request: ChatRequest):
        """Chat sincrona (senza SSE) per client semplici."""
        return await _process_chat(app, request)

    # ── Upload documenti ────────────────────────────────────────

    @app.post("/api/documents/upload", response_model=UploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        """Carica e indicizza un documento."""
        # Validazione tipo
        allowed = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp"}
        suffix = Path(file.filename or "unknown").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Formato non supportato",
                    "detail": f"Formati accettati: {', '.join(sorted(allowed))}",
                    "suggestion": f"Il file '{file.filename}' ha estensione '{suffix}'.",
                },
            )

        # Check duplicato: se il file esiste già, rimuovilo prima
        existing_files = app.state.indexer.list_source_files()
        was_duplicate = False
        if file.filename in existing_files:
            was_duplicate = True
            logger.info(f"🔄 File '{file.filename}' già presente, sostituisco con nuova versione")
            app.state.indexer.delete_by_source(file.filename)

        # Salva in temp
        temp_path = settings.TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
        try:
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Pipeline: load → chunk → embed → index
            pages = app.state.loader.load(str(temp_path))
            # Sovrascrivi source_file con il nome originale
            for p in pages:
                p.source_file = file.filename or "unknown"
            chunks = app.state.chunker.chunk_pages(pages)
            if chunks:
                texts = [c.text for c in chunks]
                embeddings = app.state.embedder.embed_documents(texts)
                chunk_ids = [c.chunk_id for c in chunks]
                metadatas = [
                    {"source_file": c.source_file, "page_number": c.page_number, "chunk_index": c.chunk_index}
                    for c in chunks
                ]
                app.state.indexer.add_chunks(chunk_ids, texts, embeddings.tolist(), metadatas)
            app.state.search_engine.rebuild_bm25_index()

            action = "sostituito" if was_duplicate else "indicizzato"
            logger.info(f"📄 {'🔄' if was_duplicate else '✅'} '{file.filename}': {len(pages)} pagine, {len(chunks)} chunks ({action})")

            msg = (
                f"Documento aggiornato (era già presente): {len(chunks)} frammenti ri-indicizzati."
                if was_duplicate
                else f"Documento indicizzato con successo: {len(chunks)} frammenti creati."
            )

            return UploadResponse(
                filename=file.filename or "unknown",
                file_type=suffix.lstrip("."),
                pages=len(pages),
                chunks=len(chunks),
                message=msg,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ── CRUD documenti ──────────────────────────────────────────

    @app.get("/api/documents", response_model=DocumentListResponse)
    async def list_documents():
        """Lista documenti indicizzati."""
        files = app.state.indexer.list_source_files()
        docs = []
        for fname in sorted(files):
            info = app.state.indexer.get_document_info(fname)
            docs.append(DocumentInfo(
                filename=fname,
                file_type=Path(fname).suffix.lstrip("."),
                chunks=info.get("chunk_count", 0),
                indexed_at=info.get("indexed_at"),
            ))
        return DocumentListResponse(documents=docs, total=len(docs))

    @app.delete("/api/documents/{filename}")
    async def delete_document(filename: str):
        """Elimina un documento dall'indice."""
        app.state.indexer.delete_by_source(filename)
        app.state.search_engine.rebuild_bm25_index()
        logger.info(f"🗑️ Documento '{filename}' rimosso dall'indice")
        return {"message": f"Documento '{filename}' eliminato con successo."}

    # ── Vision ──────────────────────────────────────────────────

    @app.post("/api/vision/analyze", response_model=VisionResponse)
    async def analyze_image(
        file: UploadFile = File(...),
        depth: str = Query("standard", pattern="^(quick|standard|deep)$"),
    ):
        """Analisi immagine artistica."""
        if not app.state.vision:
            return VisionResponse(
                analysis="La funzionalità di analisi visiva non è abilitata.",
                metadata={"enabled": False},
            )

        suffix = Path(file.filename or "img.jpg").suffix.lower()
        temp_path = settings.TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())

            text, metadata = app.state.vision.analyze(str(temp_path), depth=depth)
            return VisionResponse(
                analysis=text,
                backend=metadata.get("backend", ""),
                metadata=metadata,
            )
        finally:
            if temp_path.exists():
                temp_path.unlink()

    # ── Health ──────────────────────────────────────────────────

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        """Health check."""
        uptime = time.time() - _start_time
        backends = {
            "openrouter": bool(settings.OPENROUTER_API_KEY),
            "groq": bool(settings.GROQ_API_KEY),
            "mistral": bool(settings.MISTRAL_API_KEY),
            "gemini": bool(settings.GOOGLE_API_KEY),
            "huggingface": bool(settings.HF_TOKEN),
            "deepseek": bool(settings.DEEPSEEK_API_KEY),
            "cohere": bool(settings.COHERE_API_KEY),
        }
        active = sum(1 for v in backends.values() if v)
        status = "healthy" if active >= 2 else "degraded" if active >= 1 else "unhealthy"

        storage_path = settings.STORAGE_DIR
        disk = shutil.disk_usage(str(storage_path))

        return HealthResponse(
            status=status,
            version=settings.VERSION,
            uptime_s=round(uptime, 1),
            backends=backends,
            storage={
                "total_gb": round(disk.total / (1024**3), 1),
                "free_gb": round(disk.free / (1024**3), 1),
                "documents": len(app.state.indexer.list_source_files()),
            },
        )

    # ── Stato Backend Pool ──────────────────────────────────────

    @app.get("/api/backends/status")
    async def backends_status():
        """Stato real-time di tutti i provider LLM (cooldown, latenza, success rate)."""
        from llm.backend_pool import BackendPool
        pool = BackendPool()
        return {"backends": pool.status()}

    # ── Modalità disponibili  ───────────────────────────────────

    @app.get("/api/modalita")
    async def list_modalita():
        """Lista modalità operative."""
        from config.prompts import MODALITA_LIST
        return {"modalita": MODALITA_LIST}

    # ── Chat Sessions (persistenza) ─────────────────────────────

    @app.get("/api/sessions", response_model=ChatSessionListResponse)
    async def list_sessions():
        """Lista sessioni chat salvate."""
        sessions = app.state.chat_store.list_sessions()
        return ChatSessionListResponse(
            sessions=[ChatSessionSummary(**s) for s in sessions],
            total=len(sessions),
        )

    @app.get("/api/sessions/{session_id}", response_model=ChatSessionDetail)
    async def get_session(session_id: str):
        """Recupera una sessione completa con tutti i messaggi."""
        session = app.state.chat_store.load(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sessione non trovata")
        d = session.to_dict()
        return ChatSessionDetail(**d)

    @app.post("/api/sessions")
    async def create_session(modalita: str = "generale"):
        """Crea una nuova sessione vuota."""
        from storage.chat_store import ChatSession
        session = ChatSession(modalita=modalita)
        app.state.chat_store.save(session)
        return {"session_id": session.session_id, "title": session.title}

    @app.put("/api/sessions/{session_id}/title")
    async def rename_session(session_id: str, title: str = ""):
        """Rinomina una sessione."""
        session = app.state.chat_store.load(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sessione non trovata")
        session.title = title or "Senza titolo"
        app.state.chat_store.save(session)
        return {"session_id": session_id, "title": session.title}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Elimina una sessione chat."""
        if app.state.chat_store.delete(session_id):
            return {"message": "Sessione eliminata"}
        raise HTTPException(status_code=404, detail="Sessione non trovata")

    @app.delete("/api/sessions")
    async def delete_all_sessions():
        """Elimina tutte le sessioni."""
        count = app.state.chat_store.delete_all()
        return {"message": f"{count} sessioni eliminate"}


# ── SSE streaming implementation ─────────────────────────────────

async def _chat_stream(app: FastAPI, request: ChatRequest) -> AsyncGenerator[str, None]:
    """
    Genera eventi SSE per il client.
    Eventi: thinking → sources → answer → done
    Persiste messaggi nella sessione se session_id è presente.
    """
    def sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    # Gestione sessione persistente
    from storage.chat_store import ChatSession
    session = None
    session_id = request.session_id

    if session_id:
        session = app.state.chat_store.load(session_id)
    if not session and session_id:
        # Sessione richiesta ma non trovata: creane una nuova con lo stesso ID
        session = ChatSession(session_id=session_id, modalita=request.modalita)

    # Se abbiamo una sessione, usa la history da lì (più completa) e aggiungi il messaggio utente
    if session:
        # Usa history dalla sessione se il client non ne ha inviata una propria
        if not request.history and session.messages:
            request.history = [
                {"role": m["role"], "content": m["content"]}
                for m in session.messages[-10:]
                if m.get("role") in ("user", "assistant")
            ]
        session.add_message("user", request.query)
        if len(session.messages) == 1:
            session.auto_title()

    # Step 1: Thinking
    yield sse_event("thinking", {"message": "Analizzo la domanda..."})
    await asyncio.sleep(0)  # yield control

    try:
        response = await _process_chat(app, request)

        # Step 2: Sources
        if response.citations:
            yield sse_event("sources", {
                "citations": [c.model_dump() for c in response.citations],
            })
            await asyncio.sleep(0)

        # Step 3: Answer
        yield sse_event("answer", {
            "text": response.answer,
            "backend": response.backend_used,
            "model": response.model,
            "latency_s": response.latency_s,
            "session_id": session.session_id if session else None,
        })
        await asyncio.sleep(0)

        # Step 4: Quality (optional)
        if response.quality:
            yield sse_event("quality", response.quality)
            await asyncio.sleep(0)

        # Persisti risposta assistente nella sessione
        if session:
            session.add_message("assistant", response.answer, metadata={
                "backend": response.backend_used,
                "model": response.model,
                "latency_s": response.latency_s,
                "citations": [c.model_dump() for c in response.citations] if response.citations else [],
            })
            app.state.chat_store.save(session)

    except Exception as e:
        logger.error(f"Errore durante chat stream: {e}")
        yield sse_event("error", {
            "error": "Errore durante l'elaborazione",
            "suggestion": "Riprova tra qualche istante.",
        })

    yield sse_event("done", {})


def _build_meta_answer(query: str, doc_metadata: list) -> str:
    """Build a meta answer directly from document metadata – no LLM needed."""
    q_lower = query.lower()

    # Prepare file summaries
    file_lines = []
    total_pages = 0
    total_chunks = 0
    for doc in doc_metadata:
        fname = doc.get("source_file", "sconosciuto")
        pages = doc.get("total_pages", 0)
        chunks = doc.get("chunk_count", 0)
        total_pages += pages
        total_chunks += chunks
        ext = fname.rsplit(".", 1)[-1].upper() if "." in fname else "?"
        file_lines.append(f"- **{fname}**: {pages} pagine, {chunks} chunk indicizzati (formato: {ext})")

    header = f"Hai caricato **{len(doc_metadata)} documenti** per un totale di **{total_pages} pagine**.\n\n"
    file_list = "\n".join(file_lines)

    # Answer specific meta questions
    if any(kw in q_lower for kw in ["quante pagine", "numero di pagine", "quanto è lungo"]):
        if len(doc_metadata) == 1:
            doc = doc_metadata[0]
            return f"Il documento **{doc['source_file']}** ha **{doc['total_pages']} pagine** ({doc['chunk_count']} chunk indicizzati)."
        return f"{header}Dettaglio:\n{file_list}"

    if any(kw in q_lower for kw in ["quanti documenti", "quali documenti", "quali file", "cosa ho caricato", "file caricati"]):
        return f"{header}{file_list}"

    if any(kw in q_lower for kw in ["tipo di file", "formato", "che tipo"]):
        formats = set()
        for doc in doc_metadata:
            fname = doc.get("source_file", "")
            if "." in fname:
                formats.add(fname.rsplit(".", 1)[-1].upper())
        fmt_str = ", ".join(sorted(formats)) if formats else "sconosciuto"
        return f"I documenti caricati sono in formato: **{fmt_str}**.\n\n{file_list}"

    if any(kw in q_lower for kw in ["quanti chunk", "dimensione"]):
        return f"In totale ci sono **{total_chunks} chunk** indicizzati.\n\n{file_list}"

    # Default: show all metadata
    return f"{header}{file_list}"


def _build_greeting_answer(query: str, doc_metadata: list) -> str:
    """Build a greeting response with document info — no LLM needed."""
    q_lower = query.lower().strip()

    # Choose appropriate greeting
    if any(kw in q_lower for kw in ["grazie", "thank"]):
        intro = "Prego! Sono qui per aiutarti."
    elif any(kw in q_lower for kw in ["chi sei", "presentati", "cosa sai fare", "come funzioni"]):
        intro = ("Sono **IAGestioneArte**, il tuo assistente per l'analisi di documenti "
                 "di storia dell'arte. Posso analizzare PDF, rispondere a domande specifiche, "
                 "fare riassunti e confronti tra argomenti.")
    elif any(kw in q_lower for kw in ["buongiorno"]):
        intro = "Buongiorno! Benvenuto."
    elif any(kw in q_lower for kw in ["buonasera"]):
        intro = "Buonasera! Benvenuto."
    elif any(kw in q_lower for kw in ["aiuto", "help"]):
        intro = "Certo, sono qui per aiutarti!"
    else:
        intro = "Ciao! Benvenuto."

    if not doc_metadata:
        return f"{intro} Al momento non ci sono documenti caricati. Carica un PDF per iniziare!"

    # Add document summary
    file_lines = []
    total_pages = 0
    for doc in doc_metadata:
        fname = doc.get("source_file", "sconosciuto")
        pages = doc.get("total_pages", 0)
        chunks = doc.get("chunk_count", 0)
        total_pages += pages
        file_lines.append(f"- **{fname}** ({pages} pagine, {chunks} chunk indicizzati)")

    files_text = "\n".join(file_lines)

    return (
        f"{intro} Hai caricato {len(doc_metadata)} documenti ({total_pages} pagine totali):\n\n"
        f"{files_text}\n\n"
        "Posso aiutarti a:\n"
        "- **Analizzare** l'intero contenuto di un documento\n"
        "- **Rispondere** a domande specifiche su argomenti trattati\n"
        "- **Riassumere** sezioni o l'intero documento\n"
        "- **Confrontare** temi e artisti\n\n"
        "Cosa vorresti sapere?"
    )


def _rescue_empty_answer(
    answer: str,
    citations: list,
    query: str,
) -> str:
    """Se il LLM ha risposto 'non contiene' ma ci sono citazioni, costruisce
    una risposta sintetica dai snippet delle citazioni stesse."""
    _REFUSE = "non contiene questa informazione"
    if _REFUSE not in answer.lower():
        return answer
    if not citations:
        return answer

    # Estrai parole chiave dalla query (nomi propri, sostantivi)
    import re as _re
    _STOP = {"chi", "è", "era", "cosa", "come", "dove", "quando", "perché",
             "qual", "quali", "il", "lo", "la", "le", "gli", "i", "un", "una",
             "del", "della", "dello", "dei", "degli", "delle", "di", "da", "in",
             "con", "su", "per", "tra", "fra", "sono", "questo", "questa",
             "funziona", "tecnica", "considerato", "nato", "ha", "hanno",
             "cos", "che", "tipo"}
    words = _re.findall(r"[A-Za-zÀ-ú]+", query)
    keywords = [w for w in words if w.lower() not in _STOP and len(w) > 2]

    # Costruisci risposta dai snippet
    parts = []
    kw_label = ", ".join(keywords) if keywords else query
    parts.append(f"Dai documenti caricati, ecco le informazioni disponibili su **{kw_label}**:\n")
    for cit in citations[:5]:
        snippet = getattr(cit, "text_snippet", "")
        src = getattr(cit, "source_file", "")
        page = getattr(cit, "page_number", "?")
        if snippet:
            # Prendi le prime 2 frasi significative
            sentences = _re.split(r"[.!?]\s+", snippet)
            useful = ". ".join(s.strip() for s in sentences[:2] if len(s.strip()) > 20)
            if useful:
                parts.append(f"- {useful}. [📄 {src} | p. {page}]")

    if len(parts) <= 1:
        # Non ci sono snippet utili, mantieni risposta originale
        return answer

    return "\n".join(parts)


async def _process_chat(app: FastAPI, request: ChatRequest) -> ChatResponse:
    """Pipeline RAG completa con supporto analisi comprensiva.

    Flusso:
    1. Classifica la query (comprehensive vs focused vs greeting)
    2. Se comprehensive: recupera TUTTI i chunk distribuiti su tutte le pagine
    3. Se focused: search → rerank → top-K (come prima)
    4. Build context → LLM → filter → evaluate
    """
    from retrieval.semantic_classifier import classify_semantic
    from config.prompts import COMPREHENSIVE_ANALYSIS_PROMPT

    classification = classify_semantic(request.query)
    logger.info(
        f"📊 Query classificata: type={classification.query_type}, "
        f"confidence={classification.confidence:.2f}, "
        f"needs_page_coverage={classification.needs_page_coverage}, "
        f"intents={classification.detected_intents}"
    )
    logger.info(f"📝 Testo query: {request.query[:200]}")

    system_prompt = get_system_prompt(request.modalita)
    citations_out = []
    # Per analisi comprensiva, usa max_tokens più alto
    effective_max_tokens = classification.recommended_max_tokens

    source_files = app.state.indexer.list_source_files() if request.use_rag else []

    if request.use_rag and source_files:
        # Gather document metadata for grounding
        doc_metadata = []
        for fname in source_files:
            info = app.state.indexer.get_document_info(fname)
            if info:
                doc_metadata.append(info)

        # ──────────────────────────────────────────────────
        # PATH META: DOMANDE SUL FILE (quante pagine, ecc.)
        # FAST PATH: rispondi direttamente dai metadati, senza LLM
        # ──────────────────────────────────────────────────
        if classification.query_type == "meta" and doc_metadata:
            logger.info(f"📋 Query META (fast path): rispondo con metadati ({len(doc_metadata)} files)")
            # Genera risposta direttamente dai metadati — nessuna chiamata LLM
            meta_answer = _build_meta_answer(request.query, doc_metadata)
            return ChatResponse(
                answer=meta_answer,
                citations=[],
                backend_used="metadata",
                model="direct",
                latency_s=0.0,
                quality={"confidence": 0.95, "depth": 0.8, "coverage": 1.0, "word_count": len(meta_answer.split())},
                metadata={"provider": "FastPath", "latency_s": 0.0},
            )

        # ──────────────────────────────────────────────────
        # PATH GREETING: FAST PATH — risposta template + metadati
        # ──────────────────────────────────────────────────
        elif classification.query_type == "greeting":
            logger.info(f"👋 Greeting (fast path): {len(doc_metadata)} documenti")
            greeting_answer = _build_greeting_answer(request.query, doc_metadata)
            return ChatResponse(
                answer=greeting_answer,
                citations=[],
                backend_used="greeting",
                model="direct",
                latency_s=0.0,
                quality={"confidence": 0.95, "depth": 0.5, "coverage": 1.0,
                         "word_count": len(greeting_answer.split())},
                metadata={"provider": "FastPath", "latency_s": 0.0},
            )

        # ──────────────────────────────────────────────────
        # PATH A: ANALISI COMPRENSIVA (copertura totale pagine)
        # ──────────────────────────────────────────────────
        elif classification.needs_page_coverage and source_files:
            # Seleziona il file da analizzare:
            # 1. Se l'utente nomina un file specifico, usa quello
            # 2. Altrimenti, prendi il file con più chunk (il documento principale)
            target_file = None
            q_lower = request.query.lower()
            for fname in source_files:
                fname_clean = fname.lower().replace(".pdf", "").replace(".docx", "").replace("_", " ")
                if fname_clean in q_lower or fname.lower() in q_lower:
                    target_file = fname
                    break

            if not target_file:
                # Nessun file specifico menzionato → prendi il più grande (più chunk)
                best_file = source_files[0]
                best_chunks = 0
                for fname in source_files:
                    info = app.state.indexer.get_document_info(fname)
                    n_chunks = info.get("chunk_count", 0) if info else 0
                    if n_chunks > best_chunks:
                        best_chunks = n_chunks
                        best_file = fname
                target_file = best_file

            all_chunks = app.state.indexer.get_chunks_by_source(target_file)
            file_info = app.state.indexer.get_document_info(target_file)
            total_pages = file_info.get("total_pages", 0) if file_info else 0

            logger.info(
                f"📚 Analisi comprensiva: file={target_file}, "
                f"chunk_totali={len(all_chunks)}, pagine={total_pages}"
            )

            if all_chunks:
                # Usa il system prompt arricchito con istruzioni comprensive
                system_prompt = system_prompt + "\n" + COMPREHENSIVE_ANALYSIS_PROMPT

                prompt = app.state.context_builder.build_comprehensive(
                    all_chunks=all_chunks,
                    query=request.query,
                    source_file=target_file,
                    total_pages=total_pages,
                    history=request.history,
                    system_prompt=system_prompt,
                    doc_metadata=doc_metadata,
                )

                # Citazioni: genera da tutti i chunk selezionati (distribuite per pagina)
                # Prendi pagine uniche
                pages_seen = set()
                for chunk in all_chunks:
                    page = chunk.get("page_number", 0)
                    if page not in pages_seen:
                        pages_seen.add(page)
                        citations_out.append(
                            CitationOut(
                                citation_id=f"comp_{page}",
                                source_file=target_file,
                                page_number=page,
                                text_snippet=chunk.get("text", "")[:150],
                                score=1.0,
                            )
                        )

                logger.info(
                    f"Comprensivo: {len(citations_out)} citazioni pagine, "
                    f"max_tokens={effective_max_tokens}"
                )
            else:
                # Nessun chunk trovato per il file
                prompt = app.state.context_builder.build_simple(
                    query=request.query, history=request.history
                )

        # ──────────────────────────────────────────────────
        # PATH B: RICERCA FOCUSED (top-K con reranking)
        # ──────────────────────────────────────────────────
        else:
            # Hybrid search (include key-term extraction internally)
            results = app.state.search_engine.search(
                request.query, top_k=settings.TOP_K
            )

            # Rerank — usa la query con termini chiave per un matching più preciso
            if results:
                from retrieval.hybrid_search import HybridSearchEngine
                clean_query = HybridSearchEngine._extract_key_terms(request.query)
                if clean_query.lower() != request.query.lower() and len(clean_query) >= 3:
                    results_orig = app.state.reranker.rerank(
                        request.query, results, top_k=settings.RERANK_TOP_K * 2
                    )
                    results_clean = app.state.reranker.rerank(
                        clean_query, results, top_k=settings.RERANK_TOP_K * 2
                    )
                    merged = {}
                    for r in results_orig + results_clean:
                        if r.chunk_id not in merged or r.score > merged[r.chunk_id].score:
                            merged[r.chunk_id] = r
                    results = sorted(merged.values(), key=lambda x: x.score, reverse=True)[
                        :settings.RERANK_TOP_K
                    ]
                    logger.info(
                        f"Rerank dual-query: orig={len(results_orig)}, "
                        f"clean={len(results_clean)}, merged={len(results)}"
                    )
                else:
                    results = app.state.reranker.rerank(
                        request.query, results, top_k=settings.RERANK_TOP_K
                    )

            # Build context — solo se ci sono risultati rilevanti dopo il reranking
            if results:
                prompt = app.state.context_builder.build(
                    query=request.query,
                    search_results=results,
                    history=request.history,
                    doc_metadata=doc_metadata,
                )
                citations = app.state.citation_manager.build_citations(results)
                citations_out = [
                    CitationOut(
                        citation_id=c.citation_id,
                        source_file=c.source_file,
                        page_number=c.page_number,
                        text_snippet=c.text_snippet[:200],
                        score=c.score,
                    )
                    for c in citations
                ]
                logger.info(
                    f"RAG focused: {len(results)} risultati, {len(citations_out)} citazioni"
                )
            else:
                logger.info(
                    "Nessun risultato rilevante dal reranker, uso build_simple"
                )
                prompt = app.state.context_builder.build_simple(
                    query=request.query, history=request.history
                )
    else:
        # Greeting o no RAG — fast path
        if classification.query_type == "greeting":
            doc_metadata = []
            if source_files:
                for fname in source_files:
                    info = app.state.indexer.get_document_info(fname)
                    if info:
                        doc_metadata.append(info)
            greeting_answer = _build_greeting_answer(request.query, doc_metadata)
            return ChatResponse(
                answer=greeting_answer,
                citations=[],
                backend_used="greeting",
                model="direct",
                latency_s=0.0,
                quality={"confidence": 0.95, "depth": 0.5, "coverage": 1.0,
                         "word_count": len(greeting_answer.split())},
                metadata={"provider": "FastPath", "latency_s": 0.0},
            )
        else:
            prompt = app.state.context_builder.build_simple(
                query=request.query, history=request.history
            )

    # LLM completion — con max_tokens adattato al tipo di query
    # Smart routing: scegli la catena backend migliore per il tipo di query
    routing_hint = None
    if classification.query_type in ("meta", "greeting"):
        routing_hint = "speed"
    elif classification.needs_page_coverage:
        routing_hint = "large_context"
    elif classification.query_type == "focused":
        routing_hint = "reasoning"

    # Crea un orchestrator temporaneo con i token elevati se necessario
    if effective_max_tokens != settings.LLM_MAX_TOKENS:
        from llm.orchestrator import LLMOrchestrator
        llm_instance = LLMOrchestrator(max_tokens=effective_max_tokens)
        logger.info(f"LLM con max_tokens elevati: {effective_max_tokens}")
    else:
        llm_instance = app.state.llm

    # Tronca il contesto comprehensive se troppo grande per i provider
    if routing_hint == "large_context":
        estimated_tokens = len(prompt.split()) * 1.33
        if estimated_tokens > MAX_COMPREHENSIVE_CONTEXT_TOKENS:
            words = prompt.split()
            max_words = int(MAX_COMPREHENSIVE_CONTEXT_TOKENS / 1.33)
            prompt = ' '.join(words[:max_words])
            logger.info(f"📏 Contesto troncato: {len(words)} → {max_words} parole "
                        f"({estimated_tokens:.0f} → {MAX_COMPREHENSIVE_CONTEXT_TOKENS} token stimati)")

    llm_response = await asyncio.to_thread(
        llm_instance.complete, prompt, system_prompt,
        None,  # on_attempt callback
        routing_hint,
    )

    # Italian filter
    filtered_text = app.state.italian_filter.filter(llm_response.text)

    # ── Rescue: se il LLM dice "non contiene" ma ci sono citazioni, usa i snippet ──
    filtered_text = _rescue_empty_answer(filtered_text, citations_out, request.query)

    # Quality evaluation
    quality = app.state.quality_eval.evaluate(filtered_text, request.query)

    return ChatResponse(
        answer=filtered_text,
        citations=citations_out,
        backend_used=llm_response.backend_used,
        model=llm_response.model,
        latency_s=llm_response.latency,
        quality={
            "confidence": quality.confidence,
            "depth": quality.depth_score,
            "coverage": quality.coverage_score,
            "word_count": quality.word_count,
        },
        metadata=llm_response.metadata,
    )


# ── Static files mount ──────────────────────────────────────────

def _mount_static(app: FastAPI) -> None:
    """Monta file statici UI e serve index.html come default."""

    ui_dir = settings.PROJECT_ROOT / "ui"
    if ui_dir.exists():
        app.mount("/css", StaticFiles(directory=str(ui_dir / "css")), name="css")
        app.mount("/js", StaticFiles(directory=str(ui_dir / "js")), name="js")

        @app.get("/", response_class=HTMLResponse)
        async def serve_index():
            index_path = ui_dir / "index.html"
            if index_path.exists():
                return index_path.read_text(encoding="utf-8")
            return "<h1>IAGestioneArte</h1><p>UI non trovata.</p>"
    else:
        @app.get("/", response_class=HTMLResponse)
        async def serve_fallback():
            return (
                f"<h1>{settings.APP_NAME}</h1>"
                f"<p>v{settings.VERSION} — API disponibile su /docs</p>"
            )
