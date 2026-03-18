"""
Query classifier: determina il tipo di query per ottimizzare il retrieval.

Classificazioni (in ordine di priorità di analisi):
1. "comprehensive": analisi completa del documento (copertura totale pagine)
2. "meta": domanda sul file stesso (pagine, tipo, nome)
3. "focused": domanda specifica su un argomento (precisione, top-k ristretto)
4. "greeting": SOLO saluto puro, senza alcuna richiesta informativa

PRINCIPIO CHIAVE: Se una query contiene ANCHE SOLO UNA richiesta informativa
(analizza, riassumi, quante pagine, ecc.), NON è mai un greeting — anche se
inizia con "ciao" o "buongiorno".
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Risultato della classificazione di una query."""
    query_type: str          # "comprehensive", "focused", "meta", "greeting"
    confidence: float        # 0.0 - 1.0
    recommended_top_k: int   # Chunk consigliati da recuperare
    recommended_max_tokens: int  # Token output raccomandati per il LLM
    needs_page_coverage: bool    # Se serve copertura distribuita su tutte le pagine
    detected_intents: list = None  # Intenti rilevati nella query

    def __post_init__(self):
        if self.detected_intents is None:
            self.detected_intents = []


# ── SEGNALI DI INTENTO ──────────────────────────────────────────
# Ogni segnale ha (pattern, tipo, peso). I pesi determinano la classificazione finale.

# Segnali che indicano RICHIESTA COMPRENSIVA (analisi completa)
# I pattern usano stem verbali con \w* per coprire: analizza/analizzare/analizzami/analizzalo/ecc.
_COMPREHENSIVE_SIGNALS = [
    # ANALISI — anali[zs]+\w* copre: analizza/analizzare/analizzami/analisami/analizzalo
    (r"anali[zs]+\w*\s+(il\s+)?(file|documento|pdf|doc|tutto|completamente|questo)", "comprehensive", 2.0),
    (r"anali[zs]+\w*\s+l[''']intero", "comprehensive", 2.5),
    # RIASSUNTO — riassumi/riassumere/riassumimi + "un/il riassunto"
    (r"\b(riassunt|riassum)\w*\b", "comprehensive", 2.0),
    (r"riassunto\s+(completo|generale|totale|del)", "comprehensive", 2.5),
    # SINTESI / PANORAMICA
    (r"\bsintesi\b", "comprehensive", 2.0),
    (r"panoramica", "comprehensive", 2.0),
    # PARLAMI DI TUTTO / DIMMI TUTTO
    (r"parlami\s+(del\s+contenuto|di\s+tutto|di\s+questo)", "comprehensive", 2.0),
    (r"dimmi\s+tutto|tutte\s+le\s+informazioni", "comprehensive", 2.5),
    (r"ogni\s+(dettaglio|informazione|argomento|sezione|tema)", "comprehensive", 2.0),
    # INTERO DOCUMENTO / TUTTO IL CONTENUTO
    (r"(intero\s+documento|tutto\s+il\s+(contenuto|file|documento|testo|pdf))", "comprehensive", 2.5),
    # COSA CONTIENE / DI COSA PARLA
    (r"cosa\s+contiene", "comprehensive", 2.0),
    (r"di\s+(che\s+)?cosa\s+(parla|tratta)", "comprehensive", 2.0),
    # ARGOMENTI / TEMI
    (r"(argomenti|temi)\s+(principali|trattati|presenti|del)", "comprehensive", 2.0),
    # STRUTTURA / INDICE / SEZIONI / CAPITOLI (merged)
    (r"(struttura|indice|sezioni|capitoli)\s+(del|dell[''']|principali|presenti|dei|degli)", "comprehensive", 1.5),
    # PER INTERO / DALL'INIZIO ALLA FINE
    (r"dall[''']inizio\s+alla\s+fine|per\s+intero", "comprehensive", 2.5),
    (r"elenco\s+(completo|di\s+tutti)", "comprehensive", 2.0),
    (r"(spieg|descriv)\w*\s+(tutto|l[''']intero|il\s+contenuto)", "comprehensive", 2.0),
    # MODAL VERBS + azione: devi/puoi/potresti/sarebbe possibile + analizzare/riassumere
    (r"(devi|voglio|vorrei|puoi)\s+\w*\s*anali[zs]+\w*", "comprehensive", 2.0),
    (r"(devi|voglio|vorrei|puoi)\s+\w*\s*(un|il|una)\s+riassunto", "comprehensive", 2.0),
    (r"(potresti|potreste|sarebbe\s+possibile|(?:è|e)\s+possibile|vorresti)\s+\w*(anali[zs]+|riassum|fare|farne|spieg|descriv|illustr)\w*", "comprehensive", 2.0),
    (r"mi\s+faresti\s+(un|una|il|la|l['''\u2019])\s*\w*(riassunto|analisi|sintesi|panoramica)", "comprehensive", 2.0),
    # IMPERATIVO CORTESE: fammi/fanne
    (r"(fammi|fanne)\s+\w{0,4}['''\u2019]?\s*(riassunto|analisi|sintesi|panoramica)", "comprehensive", 2.0),
    # NEGAZIONE: "non analizzare" è richiesta informativa (non greeting)
    (r"non\s+(anali[zs]+|riassum|spieg|descriv)\w*", "focused", 1.0),
    # ANALISI come sostantivo: "un'analisi", "fare analisi", "analisi dell'intero"
    (r"\banalisi\s+(dell[''\u2019]|del|complet|general|total)", "comprehensive", 2.0),
    (r"un[''\u2019]?\s*analisi", "comprehensive", 1.5),
]

# Segnali che indicano domanda META (sul file, non sul contenuto)
_META_SIGNALS = [
    (r"quante\s+pagine", "meta", 3.0),
    (r"(che|quali?)\s+(file|document[oi])\s+(ho|sono|hai)", "meta", 3.0),
    (r"document[oi]\s+caricat[oi]", "meta", 3.0),
    (r"tipo\s+di\s+file", "meta", 2.5),
    (r"nome\s+del\s+file", "meta", 2.5),
    (r"formato\s+del", "meta", 2.0),
    (r"dimensione\s+del", "meta", 2.0),
    (r"quanti\s+chunk", "meta", 2.0),
    (r"(peso|size|dimensione)\s+del\s+(file|documento)", "meta", 2.0),
    (r"quanto\s+è\s+(lungo|lunga|grande|pesante)\s+(il|la|lo)\s+(file|pdf|documento)", "meta", 2.5),
    (r"quanti\s+(document[oi]|file)\b", "meta", 2.5),
]

# Segnali che indicano DOMANDA SPECIFICA (focused)
_FOCUSED_SIGNALS = [
    (r"(cos[''']è|che\s+cos[''']è|cosa\s+significa)", "focused", 1.5),
    (r"(chi\s+è|chi\s+era|chi\s+sono|chi\s+erano)", "focused", 1.5),
    (r"(spiega|descrivi|illustra)\s+", "focused", 1.0),
    (r"(differenz[ae]|confronta|paragona)", "focused", 1.5),
    (r"(quando|dove|perché|come)\s+", "focused", 1.0),
    (r"(quale|quali)\s+", "focused", 1.0),
    (r"(elenca|lista)\s+", "focused", 1.0),
    (r"(parla(mi)?|dimmi)\s+(di|del|della|delle|degli|dei)\s+\w+", "focused", 1.5),
]

# Pattern per saluto PURO (solo questi, senza altra richiesta)
_GREETING_PATTERNS = [
    r"^(ciao|buongiorno|buonasera|salve|hey|saluti|ehila|ehilà)[\s!.,?]*$",
    r"^grazie(\s+(mille|tante|infinite))?[\s!.,?]*$",
    r"^come\s+(stai|va)[\s!?]*$",
    r"^tutto\s+bene[\s!?]*$",
    r"^(ciao|buongiorno|buonasera|salve)[\s,!.]*come\s+(stai|va)[\s!?]*$",
    r"^(ciao|buongiorno|buonasera|salve)[\s,!.]*tutto\s+bene[\s!?]*$",
    r"^chi\s+sei[\s!?]*$",
    r"^cosa\s+sai\s+fare[\s!?]*$",
    r"^presentati[\s!?]*$",
    r"^aiuto[\s!?]*$",
]

# Frasi che SEMBRANO focused (matchano come/dove/quando) ma sono saluti
_GREETING_PHRASES = [
    r"come\s+stai", r"come\s+va", r"tutto\s+bene",
    r"come\s+ti\s+chiami", r"che\s+fai", r"come\s+funzioni",
]

# Parole che indicano SEMPRE una richiesta informativa (mai un puro greeting)
# Include stem verbali per coprire coniugazioni: analizza→analizzare/analizzami/ecc.
_INFORMATIVE_KEYWORDS = [
    "analizz", "analisa", "analis", "riassum", "riassunto", "sintesi",
    "spieg", "descriv", "pagine", "pagina", "file", "documento", "pdf",
    "contenuto", "elenca", "trova", "cerca", "confronta", "differenza",
    "cos'è", "chi è", "significa", "definisci", "illustra",
    "argomenti", "temi", "sezioni", "capitoli", "indice",
    "parlami", "dimmi", "racconta", "approfondisci", "panoramica",
    "domande", "esame", "verifica", "test", "quiz",
    "devi", "voglio", "vorrei", "fammi", "fai", "fanne", "farne",
    "potresti", "potreste", "sarebbe possibile", "è possibile",
    "faresti", "vorresti", "non analizz", "non riassum",
]


def _strip_greeting_prefix(query: str) -> str:
    """
    Rimuove il saluto iniziale per analizzare il VERO contenuto della query.
    "ciao analizza il file" → "analizza il file"
    "buongiorno, quante pagine ha?" → "quante pagine ha?"
    """
    # Pattern per saluti iniziali da rimuovere
    prefixes = [
        r"^(ciao|buongiorno|buonasera|salve|hey|ehilà?|saluti)[\s,!.;:]+",
        r"^(ciao\s+a\s+tutti?)[\s,!.;:]+",
    ]
    result = query.strip()
    for pattern in prefixes:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    return result


def _has_informative_content(query: str) -> bool:
    """Verifica se la query contiene QUALSIASI richiesta informativa."""
    q_lower = query.lower()
    for keyword in _INFORMATIVE_KEYWORDS:
        if keyword in q_lower:
            return True
    return False


def _compute_scores(query: str) -> dict:
    """
    Calcola punteggi per ogni tipo di classificazione.
    Analizza la query SENZA il saluto iniziale.
    """
    core_query = _strip_greeting_prefix(query)
    q_lower = core_query.lower().strip() if core_query else query.lower().strip()

    scores = {"comprehensive": 0.0, "meta": 0.0, "focused": 0.0, "greeting": 0.0}
    intents = []

    # Calcola punteggi comprehensive
    for pattern, _, weight in _COMPREHENSIVE_SIGNALS:
        if re.search(pattern, q_lower):
            scores["comprehensive"] += weight
            intents.append(("comprehensive", pattern))

    # Calcola punteggi meta
    for pattern, _, weight in _META_SIGNALS:
        if re.search(pattern, q_lower):
            scores["meta"] += weight
            intents.append(("meta", pattern))

    # Calcola punteggi focused (escludendo frasi di saluto mascherato)
    is_greeting_phrase = any(re.search(p, q_lower) for p in _GREETING_PHRASES)
    for pattern, _, weight in _FOCUSED_SIGNALS:
        if re.search(pattern, q_lower):
            # "come stai" non è una domanda focused, è un saluto
            if is_greeting_phrase and pattern == r"(quando|dove|perché|come)\s+":
                continue
            scores["focused"] += weight
            intents.append(("focused", pattern))

    # Greeting: controlla sia la query originale che quella strippata
    original_lower = query.lower().strip()
    for pattern in _GREETING_PATTERNS:
        if re.search(pattern, original_lower) or re.search(pattern, q_lower):
            scores["greeting"] += 3.0
            intents.append(("greeting", pattern))
            break  # Un solo match greeting è sufficiente

    # Se la query è composta SOLO da frasi di saluto (come stai, tutto bene, ecc.)
    # e non contiene keyword informative, conta come greeting
    if is_greeting_phrase and scores["greeting"] == 0:
        # Rimuovi tutte le greeting phrases e vedi se resta qualcosa di sostanziale
        remaining = q_lower
        for gp in _GREETING_PHRASES:
            remaining = re.sub(gp, "", remaining)
        remaining = re.sub(r"[\s,!?.;:]+", " ", remaining).strip()
        if len(remaining) < 3:  # Solo punteggiatura/spazi residui
            scores["greeting"] += 2.0
            intents.append(("greeting", "composite_greeting_phrase"))

    return scores, intents


def classify_query(query: str) -> QueryClassification:
    """
    Classifica una query con analisi INTELLIGENTE multi-segnale.

    PRINCIPIO: L'intento informativo vince SEMPRE sul saluto.
    "ciao analizza il file" → comprehensive (NON greeting)
    "quante pagine ha il doc?" → meta
    "ciao" → greeting

    Se la query contiene più intenti (es. "analizza il file, quante pagine ha?"),
    viene scelta la classificazione col punteggio più alto. Se comprehensive + meta
    sono entrambi presenti, si usa "comprehensive" con meta risolto nel prompt.

    Returns:
        QueryClassification con tipo, fiducia e parametri consigliati
    """
    q_lower = query.lower().strip()

    # Se la query contiene qualsiasi keyword informativa, NON è mai un greeting
    has_info = _has_informative_content(query)

    # Calcola punteggi per tutti i tipi
    scores, intents = _compute_scores(query)

    # REGOLA CHIAVE: se ci sono segnali informativi, greeting punteggio → 0
    if has_info:
        scores["greeting"] = 0.0

    # Se comprehensive + meta entrambi presenti, comprehensive vince
    # (la risposta comprehensive includerà anche i metadati)
    if scores["comprehensive"] > 0 and scores["meta"] > 0:
        scores["comprehensive"] += scores["meta"] * 0.5

    # Trova il tipo con punteggio massimo
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Se nessun segnale forte e la query non è un greeting puro → focused
    if best_score <= 0:
        # Short query con parola di contenuto (nome proprio, termine tecnico)
        # → è un follow-up o domanda implicita, non un greeting
        words = [w.strip("?!.,;:") for w in q_lower.split() if len(w.strip("?!.,;:")) > 1]
        # Congiunzioni/preposizioni pure che NON indicano contenuto
        _FILLER = {"e", "o", "a", "di", "da", "in", "con", "su", "per", "tra",
                   "fra", "il", "lo", "la", "le", "gli", "i", "un", "una", "ma",
                   "se", "che", "non", "mi", "ti", "ci", "si", "vi", "ne",
                   "al", "del", "nel", "dal", "sul", "col"}
        content_words = [w for w in words if w not in _FILLER]
        has_capitalized = any(w[0].isupper() for w in query.split()
                             if len(w) > 1 and w[0].isupper())

        if content_words or has_capitalized:
            best_type = "focused"
            best_score = 1.0
        elif len(words) >= 3:
            best_type = "focused"
            best_score = 1.0
        else:
            # Query troppo corta senza contenuto → greeting
            best_type = "greeting"
            best_score = 2.0

    # Calcola confidence normalizzata
    total_score = sum(scores.values()) or 1.0
    confidence = min(best_score / max(total_score, 1.0), 0.98)
    confidence = max(confidence, 0.5)  # minimo 0.5

    # Intent descriptions per logging
    detected = [f"{t}:{p}" for t, p in intents]

    # Configura parametri in base al tipo
    configs = {
        "comprehensive": {
            "recommended_top_k": 30,
            "recommended_max_tokens": 4096,
            "needs_page_coverage": True,
        },
        "meta": {
            "recommended_top_k": 0,
            "recommended_max_tokens": 1024,
            "needs_page_coverage": False,
        },
        "focused": {
            "recommended_top_k": 10,
            "recommended_max_tokens": 2048,
            "needs_page_coverage": False,
        },
        "greeting": {
            "recommended_top_k": 0,
            "recommended_max_tokens": 1024,
            "needs_page_coverage": False,
        },
    }

    cfg = configs.get(best_type, configs["focused"])

    return QueryClassification(
        query_type=best_type,
        confidence=round(confidence, 2),
        detected_intents=detected,
        **cfg,
    )
