"""
Logging configurazione: log strutturati con rotazione file + console.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import settings


def setup_logging() -> None:
    """
    Configura logging con:
    - Console: INFO (formattato leggibile)
    - File: DEBUG con rotazione (5 MB * 3 backup)
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    # Formato
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # File handler con rotazione
    log_dir = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    root.addHandler(file_handler)

    # Silenzia log verbose di terze parti
    for noisy in [
        "httpcore", "httpx", "urllib3", "chromadb",
        "sentence_transformers", "transformers", "huggingface_hub",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Silenzia completamente errore telemetrica ChromaDB (harmless posthog compat issue)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

    logging.info(f"📝 Logging configurato — file: {log_file}")
