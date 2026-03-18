"""
Classificatore semantico: usa Ollama llama3.2:3b per classificare le query.

Due livelli:
1. Ollama semantico (timeout 3s) — capisce intento anche in frasi non viste
2. Regex fallback — se Ollama è offline o lento, usa il classificatore originale

Ritorna lo stesso QueryClassification del classificatore regex per compatibilità.
"""

import logging
import requests
import json
import time
from typing import Optional

from retrieval.query_classifier import classify_query, QueryClassification

logger = logging.getLogger(__name__)

# Prompt di classificazione per il modello locale
_CLASSIFICATION_PROMPT = """Classifica la seguente query utente in UNA di queste categorie:

- "greeting": saluto puro senza richieste informative (es. "ciao", "buongiorno")
- "meta": domanda sul file/documento stesso (pagine, tipo, nome, lista documenti)
- "comprehensive": richiesta di analisi completa, riassunto, panoramica generale
- "focused": domanda specifica su un argomento, artista, periodo, opera

REGOLA FONDAMENTALE: Se la query contiene ANCHE SOLO UNA richiesta informativa
(analizza, riassumi, quante pagine, chi era, cosa sai di...), NON è mai "greeting".

Rispondi con SOLO il JSON, nient'altro:
{{"type": "<categoria>", "confidence": <0.0-1.0>}}

Query: """

# Impostazioni raccomandate per ciascun tipo
_TYPE_SETTINGS = {
    "greeting": {
        "top_k": 0,
        "max_tokens": 150,
        "needs_page_coverage": False,
    },
    "meta": {
        "top_k": 0,
        "max_tokens": 300,
        "needs_page_coverage": False,
    },
    "focused": {
        "top_k": 8,
        "max_tokens": 1200,
        "needs_page_coverage": False,
    },
    "comprehensive": {
        "top_k": 40,
        "max_tokens": 4000,
        "needs_page_coverage": True,
    },
}

# Cache di stato Ollama per non riprovare ad ogni query se è offline
_ollama_available: Optional[bool] = None
_ollama_last_check: float = 0.0
_OLLAMA_RECHECK_INTERVAL = 60.0  # Ricontrolla ogni 60s se era offline


def _check_ollama() -> bool:
    """Verifica che Ollama sia raggiungibile e llama3.2:3b sia disponibile."""
    global _ollama_available, _ollama_last_check
    now = time.time()
    if _ollama_available is not None and (now - _ollama_last_check) < _OLLAMA_RECHECK_INTERVAL:
        return _ollama_available
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            _ollama_available = any("llama3.2:3b" in m for m in models)
        else:
            _ollama_available = False
    except Exception:
        _ollama_available = False
    _ollama_last_check = now
    if not _ollama_available:
        logger.debug("Ollama llama3.2:3b non disponibile, usando regex fallback")
    return _ollama_available


def _classify_with_ollama(query: str, timeout: float = 3.0) -> Optional[dict]:
    """
    Chiede a Ollama llama3.2:3b di classificare la query.
    Ritorna {"type": "...", "confidence": ...} o None se fallisce.
    """
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": _CLASSIFICATION_PROMPT + query,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 50,
                },
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None

        raw = resp.json().get("response", "").strip()
        # Estrai JSON dalla risposta (potrebbe avere testo extra)
        # Cerca il primo { e l'ultimo }
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx == -1 or end_idx == -1:
            logger.debug(f"Ollama risposta non-JSON: {raw[:100]}")
            return None

        json_str = raw[start_idx:end_idx + 1]
        parsed = json.loads(json_str)

        qtype = parsed.get("type", "").lower().strip()
        confidence = float(parsed.get("confidence", 0.5))

        if qtype not in _TYPE_SETTINGS:
            logger.debug(f"Ollama tipo non valido: {qtype}")
            return None

        return {"type": qtype, "confidence": min(1.0, max(0.0, confidence))}

    except (requests.Timeout, requests.ConnectionError):
        logger.debug("Ollama timeout/connection error in classify")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.debug(f"Ollama parse error: {e}")
        return None
    except Exception as e:
        logger.warning(f"Ollama classify unexpected error: {e}")
        return None


def classify_semantic(query: str) -> QueryClassification:
    """
    Classificatore ibrido a due livelli:
    1. Regex FIRST (0ms, 40/40 ground truth) — per pattern noti
    2. Ollama semantico SOLO per query ambigue (regex confidence < 0.7)

    Questo approccio garantisce:
    - Latenza zero per greeting/meta (fast-path ≤5s)
    - Copertura semantica per query mai viste
    """
    # Livello 1: Regex (sempre primo — velocità garantita)
    regex_result = classify_query(query)

    # Se regex è sicuro (confidence ≥ 0.7), usa direttamente — zero overhead
    if regex_result.confidence >= 0.7:
        logger.info(
            f"📏 Regex classifier (fast): {regex_result.query_type} "
            f"(confidence={regex_result.confidence:.2f})"
        )
        return regex_result

    # Livello 2: Ollama semantico per query ambigue
    if _check_ollama():
        start = time.time()
        result = _classify_with_ollama(query)
        latency_ms = (time.time() - start) * 1000

        if result:
            qtype = result["type"]
            confidence = result["confidence"]
            settings = _TYPE_SETTINGS[qtype]

            logger.info(
                f"🧠 Semantic classifier: {qtype} "
                f"(confidence={confidence:.2f}, {latency_ms:.0f}ms) "
                f"[regex was {regex_result.query_type}@{regex_result.confidence:.2f}]"
            )
            return QueryClassification(
                query_type=qtype,
                confidence=confidence,
                recommended_top_k=settings["top_k"],
                recommended_max_tokens=settings["max_tokens"],
                needs_page_coverage=settings["needs_page_coverage"],
                detected_intents=[f"semantic:{qtype}"],
            )
        else:
            logger.debug(
                f"Ollama classify fallito ({latency_ms:.0f}ms), usando regex"
            )

    # Fallback: regex comunque (qualsiasi confidence)
    logger.info(
        f"📏 Regex fallback: {regex_result.query_type} "
        f"(confidence={regex_result.confidence:.2f})"
    )
    return regex_result
