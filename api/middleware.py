"""
Middleware FastAPI: error handling, CORS, request logging.
Nessun stack trace esposto all'utente.
"""

import logging
import time
import traceback
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Configura tutti i middleware sull'app FastAPI."""

    # CORS (permetti tutto in dev, restringi in produzione)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
        """Log ogni richiesta con timing."""
        start = time.time()
        method = request.method
        path = request.url.path

        # Skip health check dal log
        if path == "/api/health":
            return await call_next(request)

        logger.info(f"→ {method} {path}")

        try:
            response = await call_next(request)
            elapsed = time.time() - start
            logger.info(f"← {method} {path} [{response.status_code}] {elapsed:.2f}s")
            return response

        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"✗ {method} {path} [{elapsed:.2f}s] {e}")
            raise

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Handler globale errori: nessuno stack trace all'utente.
        Solo messaggi umani in italiano.
        """
        logger.error(f"Errore non gestito: {exc}\n{traceback.format_exc()}")

        # Mappa errori → messaggi umani
        error_map = {
            "FileNotFoundError": (
                "File non trovato",
                "Il file richiesto non è disponibile.",
                "Verifica il nome del file e riprova.",
            ),
            "PermissionError": (
                "Errore di permessi",
                "Non hai i permessi per accedere a questa risorsa.",
                "Contatta l'amministratore.",
            ),
            "ValueError": (
                "Dati non validi",
                "I dati inviati non sono nel formato corretto.",
                "Controlla i parametri e riprova.",
            ),
            "ConnectionError": (
                "Errore di connessione",
                "Impossibile connettersi al servizio richiesto.",
                "Riprova tra qualche istante.",
            ),
            "TimeoutError": (
                "Tempo scaduto",
                "La richiesta ha impiegato troppo tempo.",
                "Riprova con una domanda più breve.",
            ),
        }

        exc_type = type(exc).__name__
        error, detail, suggestion = error_map.get(
            exc_type,
            (
                "Errore interno",
                "Si è verificato un problema durante l'elaborazione.",
                "Riprova tra qualche istante. Se il problema persiste, ricarica la pagina.",
            ),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": error,
                "detail": detail,
                "suggestion": suggestion,
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Risorsa non trovata",
                "detail": f"'{request.url.path}' non esiste.",
                "suggestion": "Verifica l'URL e riprova.",
            },
        )

    @app.exception_handler(422)
    async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Dati non validi",
                "detail": "I parametri inviati non rispettano il formato richiesto.",
                "suggestion": "Controlla la domanda e riprova.",
            },
        )
