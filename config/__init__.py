"""
Configurazione centralizzata — carica config.yaml + .env
Uso: from config import settings
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent

# Carica variabili da .env
load_dotenv(PROJECT_ROOT / ".env")


def _load_yaml() -> Dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_config = _load_yaml()


class Settings:
    """Accesso centralizzato a tutta la configurazione."""

    def __init__(self, config: Dict[str, Any]):
        self._raw = config

        # --- App ---
        app = config.get("app", {})
        self.APP_NAME: str = app.get("name", "IAGestioneArte")
        self.VERSION: str = app.get("version", "1.0.0")
        self.HOST: str = app.get("host", "0.0.0.0")
        self.PORT: int = app.get("port", 8000)
        self.DEBUG: bool = app.get("debug", False)

        # --- Paths ---
        paths = config.get("paths", {})
        self.DATA_DIR: Path = PROJECT_ROOT / paths.get("data_dir", "data")
        self.STORAGE_DIR: Path = PROJECT_ROOT / paths.get("storage_dir", "storage")
        self.LOGS_DIR: Path = PROJECT_ROOT / paths.get("logs_dir", "logs")
        self.TEMP_DIR: Path = PROJECT_ROOT / paths.get("temp_dir", "temp_uploads")

        for d in [self.DATA_DIR, self.STORAGE_DIR, self.LOGS_DIR, self.TEMP_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # --- RAG ---
        rag = config.get("rag", {})
        self.CHUNK_SIZE: int = rag.get("chunk_size", 512)
        self.CHUNK_OVERLAP: int = rag.get("chunk_overlap", 50)
        self.TOP_K: int = rag.get("top_k", 10)
        self.RERANK_TOP_K: int = rag.get("rerank_top_k", 5)
        self.SCORE_THRESHOLD: float = rag.get("score_threshold", 0.15)
        self.EMBEDDING_MODEL: str = rag.get("embedding_model", "intfloat/multilingual-e5-base")
        self.RERANKER_MODEL: str = rag.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.RERANKER_PROVIDER: str = rag.get("reranker_provider", "local")

        # --- LLM ---
        llm = config.get("llm", {})
        self.LLM_TEMPERATURE: float = llm.get("temperature", 0.3)
        self.LLM_MAX_TOKENS: int = llm.get("max_tokens", 2048)
        self.LLM_DEFAULT_BACKEND: str = llm.get("default_backend", "auto")
        self.ITALIAN_ONLY: bool = llm.get("italian_only", True)

        # --- Vision ---
        vision = config.get("vision", {})
        self.VISION_ENABLED: bool = vision.get("enabled", True)
        self.OCR_ENABLED: bool = vision.get("ocr_enabled", True)

        # --- Cleanup ---
        cleanup = config.get("cleanup", {})
        self.AUTO_CLEANUP: bool = cleanup.get("auto_cleanup", True)
        self.CLEANUP_INTERVAL: int = cleanup.get("interval_seconds", 3600)
        self.TEMP_MAX_AGE_HOURS: int = cleanup.get("temp_max_age_hours", 24)
        self.LOG_MAX_AGE_DAYS: int = cleanup.get("log_max_age_days", 7)

        # --- API Keys (da .env) ---
        self.OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
        self.GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        self.MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
        self.HF_TOKEN: str = os.getenv("HF_TOKEN", "")
        self.DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
        self.COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
        self.CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
        self.SAMBANOVA_API_KEY: str = os.getenv("SAMBANOVA_API_KEY", "")
        self.OVH_API_KEY: str = os.getenv("OVH_API_KEY", "")
        self.SCALEWAY_API_KEY: str = os.getenv("SCALEWAY_API_KEY", "")
        self.GITHUB_TOKEN: str = os.getenv("GITHUB", "")
        self.NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", os.getenv("nvidia", ""))
        self.CLOUDFLARE_API_TOKEN: str = os.getenv("CLOUDFLARE_API_KEY", os.getenv("cloudflare", ""))
        self.CLOUDFLARE_ACCOUNT_ID: str = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        self.AIMLAPI_KEY: str = os.getenv("AIMLAPI_KEY", os.getenv("AIMLAPI", ""))
        
        # === Chiavi aggiunte per espansione provider ===
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
        self.XAI_API_KEY_2: str = os.getenv("XAI_API_KEY_2", "")
        self.HYPERBOLIC_API_KEY: str = os.getenv("HYPERBOLIC_API_KEY", "")
        self.FIREWORKS_API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
        self.CHUTES_API_KEY: str = os.getenv("CHUTES_API_KEY", "")
        self.TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
        self.SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
        self.STABILITY_API_KEY: str = os.getenv("STABILITY_API_KEY", "")
        self.MODELSCOPE_API_KEY: str = os.getenv("MODELSCOPE_API_KEY", "")

    @property
    def PROJECT_ROOT(self) -> Path:
        return PROJECT_ROOT


settings = Settings(_config)


def get_config() -> Settings:
    """Ritorna l'istanza di configurazione globale."""
    return settings
