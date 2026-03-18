"""
IAGestioneArte — Entry point principale.

Avvia il server FastAPI con tutti i servizi RAG integrati.
Uso: python main.py
     oppure: uvicorn main:app --host 0.0.0.0 --port 8001
"""

import sys
from pathlib import Path

# Assicura che il progetto sia nel sys.path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Expose app at module level for `uvicorn main:app`
from api.routes import create_app as _create_app

app = _create_app()


def main():
    # Import config e logging per primi
    from config import settings
    from system.logging_config import setup_logging

    setup_logging()

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Avvio {settings.APP_NAME} v{settings.VERSION}")

    # Cleanup automatico
    if settings.AUTO_CLEANUP:
        from system.cleanup import CleanupManager
        cleanup = CleanupManager()
        cleanup.start()
        logger.info("♻️ Cleanup automatico attivato")

    # Crea e avvia l'app FastAPI
    from api.routes import create_app
    import uvicorn

    app = create_app()

    logger.info(f"🌐 Server in avvio su http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📖 API docs: http://localhost:{settings.PORT}/docs")
    logger.info(f"🖥️  UI: http://localhost:{settings.PORT}/")

    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level="info" if not settings.DEBUG else "debug",
    )


if __name__ == "__main__":
    main()
