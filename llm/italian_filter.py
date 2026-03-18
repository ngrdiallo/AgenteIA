"""
Filtro italiano: assicura che le risposte siano in italiano.
PORTED FROM: AgenteIA-Production/src/advanced_reasoning_llm.py
(_detect_language_ratio, _force_italian_response, _force_italian_aggressive)
"""

import logging
import re
from typing import Set

logger = logging.getLogger(__name__)


class ItalianFilter:
    """
    Post-processing per forzare output in italiano.
    Funziona in due livelli:
    - Gentile (ratio ≤ 40%): sostituisce termini inglesi isolati
    - Aggressivo (ratio > 40%): rimuove frasi prevalentemente in inglese
    """

    # PORTED FROM: advanced_reasoning_llm.py COMMON_ENGLISH_WORDS
    COMMON_ENGLISH_WORDS: Set[str] = {
        "hello", "hi", "hey", "thank", "thanks", "yes", "no", "ok", "okay",
        "the", "and", "of", "to", "in", "is", "a", "for", "that", "with",
        "why", "what", "how", "who", "when", "where", "because", "if", "would",
        "could", "should", "can", "do", "does", "did", "have", "has", "had",
        "be", "are", "as", "from", "by", "on", "at", "this", "it", "or",
        "but", "not", "just", "only", "some", "my", "your", "his", "her",
        "good", "bad", "new", "old", "big", "small", "about", "also", "very",
        "much", "many", "more", "most", "well", "way", "may", "will", "shall",
        "than", "then", "now", "here", "there", "so", "up", "out", "into",
    }

    GENTLE_REPLACEMENTS = {
        r"\bHello\b": "Ciao",
        r"\bhi\b": "ciao",
        r"\bthanks\b": "grazie",
        r"\bpleasure\b": "piacere",
        r"\bHow are you\b": "Come stai",
        r"\bspecifically designed for\b": "specificamente progettato per",
        r"\bdesigned for\b": "progettato per",
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def filter(self, text: str) -> str:
        """Filtra il testo per garantire output in italiano."""
        if not self.enabled or not text:
            return text

        english_ratio = self._detect_language_ratio(text)

        if english_ratio <= 0.20:
            # Quasi tutto italiano, nessun intervento
            return text
        elif english_ratio <= 0.40:
            # Prevalentemente italiano con parole inglesi sparse
            return self._gentle_filter(text)
        else:
            # Troppo inglese: rimuovi frasi non italiane
            return self._aggressive_filter(text)

    def _detect_language_ratio(self, text: str) -> float:
        """Calcola la percentuale di parole inglesi nel testo."""
        words = text.lower().split()
        if not words:
            return 0.0
        english_count = sum(
            1 for w in words if w.strip(".,!?;:\"'()") in self.COMMON_ENGLISH_WORDS
        )
        return english_count / len(words)

    def _gentle_filter(self, text: str) -> str:
        """Sostituisce termini inglesi comuni con equivalenti italiani."""
        result = text
        for pattern, replacement in self.GENTLE_REPLACEMENTS.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result

    def _aggressive_filter(self, text: str) -> str:
        """Rimuove frasi prevalentemente in inglese."""
        sentences = re.split(r"[.!?]\s+", text)
        italian_sentences = []

        for sent in sentences:
            if not sent.strip():
                continue
            ratio = self._detect_language_ratio(sent)
            if ratio < 0.35:  # Mantieni frasi con meno del 35% di inglese
                italian_sentences.append(sent.strip())

        if not italian_sentences:
            # Se tutto è stato filtrato, ritorna l'originale (meglio che niente)
            logger.warning("Filtro italiano aggressivo: nessuna frase italiana trovata")
            return text

        result = ". ".join(italian_sentences)
        if not result.endswith("."):
            result += "."

        logger.debug(
            f"Filtro aggressivo: {len(sentences)} frasi → {len(italian_sentences)} mantenute"
        )
        return result
