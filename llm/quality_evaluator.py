"""
Response quality evaluator: valuta profondità e copertura delle risposte.
Decide se la risposta è adeguata o serve escalation a un modello superiore.
PORTED FROM: AgenteIA-Production/src/advanced_reasoning_llm.py (ResponseQualityEvaluator)
ENHANCED: structural analysis, citation counting, question-coverage detection.
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Risultato della valutazione qualità."""
    word_count: int
    depth_score: float       # 0.0-1.0 — profondità analitica
    coverage_score: float    # 0.0-1.0 — completezza risposta
    confidence: float        # 0.0-1.0 — media pesata dei sotto-punteggi
    needs_escalation: bool
    reason: str


class ResponseQualityEvaluator:
    """
    Valuta se una risposta LLM è sufficientemente profonda.
    Usa metriche multiple: keyword analitici, struttura, citazioni, copertura della domanda.
    Se inadeguata, segnala la necessità di escalation a un modello più capace.
    """

    # Keyword analitici per profondità ragionativa
    DEPTH_KEYWORDS = {
        "perché", "quindi", "tuttavia", "in contrasto", "d'altra parte",
        "analogamente", "in particolare", "ad esempio", "conseguentemente",
        "analizzando", "confrontando", "storicamente", "artisticamente",
        "influenza", "stile", "periodo", "corrente", "movimento", "opera",
        "confronto", "critica", "interpretazione", "significato", "contesto",
        "iconografia", "composizione", "tecnica", "prospettiva", "simbolo",
        "evidenzia", "dimostra", "suggerisce", "implica", "riflette",
        "caratterizza", "distingue", "connette", "rappresenta", "esprime",
    }

    COMPLEX_TRIGGERS = {
        "confronta", "analizza", "spiega", "collega", "saggio",
        "presentazione", "approfondisci", "critica", "valuta", "sintetizza",
        "riassumi", "panoramica", "completo", "intero", "tutto",
    }

    # Pattern strutturali che indicano risposta ben organizzata
    _STRUCTURE_PATTERNS = [
        r"^\d+[\.\)]\s",                     # Liste numerate: "1. ", "1) "
        r"^[-•]\s",                           # Liste puntate
        r"^#{1,4}\s",                         # Headers Markdown
        r"^\*\*[^*]+\*\*",                    # Bold headers
        r"\[📄[^\]]+\]",                      # Citazioni con fonte
    ]

    def evaluate(self, response: str, question: str) -> QualityScore:
        """
        Valuta qualità della risposta con metriche composite.

        Metriche:
        1. Keyword analitici (profondità ragionativa)
        2. Struttura (liste, headers, formattazione)
        3. Citazioni (riferimenti a fonti)
        4. Copertura della domanda (parole chiave della domanda nella risposta)
        5. Lunghezza relativa alla complessità

        Returns:
            QualityScore con metriche e decisione escalation
        """
        words = response.split()
        word_count = len(words)
        lines = response.split("\n")

        # ── 1. Profondità analitica (keyword) ──
        depth_hits = sum(
            1 for w in words
            if w.lower().strip(".,;:!?()\"'") in self.DEPTH_KEYWORDS
        )
        keyword_score = min(depth_hits / 6.0, 1.0)

        # ── 2. Struttura (liste, headers, bold) ──
        structure_hits = 0
        for line in lines:
            stripped = line.strip()
            for pattern in self._STRUCTURE_PATTERNS:
                if re.search(pattern, stripped, re.MULTILINE):
                    structure_hits += 1
                    break
        structure_score = min(structure_hits / 5.0, 1.0)

        # ── 3. Citazioni (📄 fonte) ──
        citation_count = len(re.findall(r"\[📄[^\]]+\]", response))
        citation_score = min(citation_count / 3.0, 1.0)

        # ── 4. Copertura domanda ──
        q_words = set(
            w.lower().strip(".,;:!?()\"'")
            for w in question.split()
            if len(w) > 3  # Ignora parole troppo corte
        )
        if q_words:
            r_lower = response.lower()
            covered = sum(1 for w in q_words if w in r_lower)
            coverage_q = covered / len(q_words)
        else:
            coverage_q = 0.5

        # ── Composizione punteggi ──
        is_complex = any(t in question.lower() for t in self.COMPLEX_TRIGGERS)
        target = 400 if is_complex else 200
        length_score = min(word_count / target, 1.0)

        # Depth: 40% keyword + 20% structure + 20% citations + 20% length
        depth_score = round(
            keyword_score * 0.4
            + structure_score * 0.2
            + citation_score * 0.2
            + length_score * 0.2,
            2,
        )

        # Coverage: 50% question coverage + 30% length + 20% structure
        coverage_score = round(
            coverage_q * 0.5 + length_score * 0.3 + structure_score * 0.2,
            2,
        )

        # Confidence: media pesata
        confidence = round((depth_score * 0.6 + coverage_score * 0.4), 2)

        # Escalation necessaria?
        needs_escalation = (
            depth_score < 0.3
            or (is_complex and word_count < 120)
            or (not is_complex and word_count < 40)
            or coverage_q < 0.3
        )

        # Motivo
        if not needs_escalation:
            reason = "Risposta adeguata"
        elif depth_score < 0.3:
            reason = (
                f"Profondità insufficiente (keywords={depth_hits}, "
                f"struttura={structure_hits}, citazioni={citation_count})"
            )
        elif coverage_q < 0.3:
            reason = f"Copertura domanda bassa ({coverage_q:.0%} parole chiave coperte)"
        elif is_complex and word_count < 120:
            reason = f"Risposta breve per domanda complessa ({word_count} parole)"
        else:
            reason = f"Risposta troppo breve ({word_count} parole)"

        return QualityScore(
            word_count=word_count,
            depth_score=depth_score,
            coverage_score=coverage_score,
            confidence=confidence,
            needs_escalation=needs_escalation,
            reason=reason,
        )
