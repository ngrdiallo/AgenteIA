#!/usr/bin/env python3
"""
Test suite per le fix dell'audit critico.
Verifica: classifier (conditional verbs, negation), quality evaluator (composite scoring),
orchestrator (timeout enforcement), token estimation consistency.

Esegui: python test_audit_fixes.py
"""

import sys
import os
import time

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════
# TEST 1: CLASSIFIER — MODAL VERBS & POLITE FORMS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: CLASSIFIER — Verbi condizionali e forme cortesi")
print("=" * 70)

from retrieval.query_classifier import classify_query

# Forme cortesi che prima non funzionavano
cortesi_comprehensive = [
    "potresti analizzare il file per favore",
    "sarebbe possibile fare un riassunto del documento",
    "è possibile analizzare tutto il file",
    "mi faresti un riassunto del documento",
    "potreste analizzare il pdf",
    "vorresti analizzare il file caricato",
    "fammi un riassunto completo",
    "fanne un'analisi dettagliata",
]

for query in cortesi_comprehensive:
    r = classify_query(query)
    test(
        f"'{query[:50]}' → comprehensive",
        r.query_type == "comprehensive",
        f"got {r.query_type} (scores via intents: {r.detected_intents[:3]})",
    )

# Negazioni — NON devono essere greeting
negations = [
    ("non analizzare questo file", "focused"),  # negazione = richiesta informativa
    ("non riassumere il documento", "focused"),
    ("non descrivere l'opera", "focused"),
]

for query, expected in negations:
    r = classify_query(query)
    test(
        f"'{query}' → {expected} (non greeting)",
        r.query_type != "greeting",
        f"got {r.query_type}",
    )

# Verifica che i test precedenti (pre-audit) funzionino ancora
pre_audit_cases = [
    ("ciao", "greeting"),
    ("buongiorno, come stai?", "greeting"),
    ("quante pagine ha il documento?", "meta"),
    ("analizza il file", "comprehensive"),
    ("devi analizzare il file. quante pagine ha. devi farne un riassunto.", "comprehensive"),
    ("parlami del Barocco", "focused"),
    ("cos'è il Rinascimento?", "focused"),
    ("chi è Bernini?", "focused"),
    ("ciao analizza il file", "comprehensive"),
    ("buongiorno, fammi un riassunto del pdf", "comprehensive"),
]

print("\n  — Regressione test pre-audit —")
for query, expected in pre_audit_cases:
    r = classify_query(query)
    test(
        f"'{query[:50]}' → {expected}",
        r.query_type == expected,
        f"got {r.query_type}",
    )

# ═══════════════════════════════════════════════════════════════
# TEST 2: QUALITY EVALUATOR — COMPOSITE SCORING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: QUALITY EVALUATOR — Metriche composite")
print("=" * 70)

from llm.quality_evaluator import ResponseQualityEvaluator

ev = ResponseQualityEvaluator()

# Risposta profonda con struttura, citazioni, ragionamento
deep_response = """
## Il Barocco: Analisi Storico-Artistica

Il Barocco rappresenta un movimento artistico fondamentale che si sviluppa nel XVII secolo.

### 1. Contesto storico
In particolare, il Barocco emerge come reazione alla Controriforma. L'influenza della
Chiesa cattolica è determinante: quindi, l'arte diventa strumento di persuasione e propaganda.

### 2. Caratteristiche stilistiche
- **Dinamismo compositivo**: La composizione barocca si caratterizza per il movimento.
- **Chiaroscuro drammatico**: La tecnica del chiaroscuro raggiunge il suo apice.
- **Prospettiva illusionistica**: Analogamente, lo spazio architettonico viene manipolato.

### 3. Artisti principali
Confrontando Bernini con Borromini, si evidenzia una differenza fondamentale:
Bernini esprime una visione classicista, tuttavia Borromini rappresenta un'interpretazione
più radicale [📄 Storia dell'Arte Moderna 13 O.pdf | p. 45].

In contrasto con il Rinascimento, il Barocco distingue tra forma e contenuto in modo
iconograficamente diverso [📄 Storia dell'Arte Moderna 13 O.pdf | p. 78].
"""

q_complex = "confronta il Barocco con il Rinascimento, analizzando artisti e stili"
score_deep = ev.evaluate(deep_response, q_complex)
test("Risposta profonda → depth > 0.5", score_deep.depth_score > 0.5,
     f"depth={score_deep.depth_score}")
test("Risposta profonda → coverage > 0.5", score_deep.coverage_score > 0.5,
     f"coverage={score_deep.coverage_score}")
test("Risposta profonda → no escalation", not score_deep.needs_escalation,
     score_deep.reason)

# Risposta superficiale — solo una frase
shallow_response = "Il Barocco è un periodo dell'arte."
score_shallow = ev.evaluate(shallow_response, q_complex)
test("Risposta superficiale → depth < 0.3", score_shallow.depth_score < 0.3,
     f"depth={score_shallow.depth_score}")
test("Risposta superficiale → needs escalation", score_shallow.needs_escalation,
     score_shallow.reason)

# Risposta media
medium_response = """Il Barocco è un movimento artistico del XVII secolo caratterizzato
da dinamismo, drammaticità e composizioni complesse. Confrontando con il Rinascimento,
il Barocco è più teatrale. L'influenza della Chiesa è fondamentale. Bernini e Borromini
sono i principali artisti. La tecnica del chiaroscuro raggiunge il suo apice nel
Barocco, in particolare con Caravaggio che utilizza la luce come strumento narrativo."""
score_medium = ev.evaluate(medium_response, q_complex)
test("Risposta media → confidence 0.2-0.8",
     0.2 <= score_medium.confidence <= 0.8,
     f"confidence={score_medium.confidence}")

# Deep > shallow
test("Deep.confidence > shallow.confidence",
     score_deep.confidence > score_shallow.confidence,
     f"{score_deep.confidence} vs {score_shallow.confidence}")

# ═══════════════════════════════════════════════════════════════
# TEST 3: ORCHESTRATOR — TIMEOUT INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: ORCHESTRATOR — Timeout infrastructure")
print("=" * 70)

from llm.orchestrator import LLMOrchestrator
import concurrent.futures

orch = LLMOrchestrator()

# Verify timeout dict exists and has sane values
test("_BACKEND_TIMEOUTS exists", hasattr(orch, '_BACKEND_TIMEOUTS'))
test("All backends have timeout",
     all(b in orch._BACKEND_TIMEOUTS for b in orch.BACKEND_CHAIN),
     f"missing: {set(orch.BACKEND_CHAIN) - set(orch._BACKEND_TIMEOUTS)}")
test("Timeouts are 10-120s",
     all(10 <= v <= 120 for v in orch._BACKEND_TIMEOUTS.values()),
     str(orch._BACKEND_TIMEOUTS))
test("Speed backends ≤ 30s",
     orch._BACKEND_TIMEOUTS["cerebras"] <= 30 and orch._BACKEND_TIMEOUTS["groq"] <= 30)
test("Gemini timeout ≤ 60s", orch._BACKEND_TIMEOUTS["gemini"] <= 60)

# Test _call_with_timeout actually enforces timeout
def slow_backend(prompt, system_prompt):
    time.sleep(5)
    return "slow", True, {"latency_s": 5}

start = time.time()
try:
    result = orch._call_with_timeout(slow_backend, "test", "", timeout_s=1)
    test("Timeout enforcement: should have raised", False, "no exception raised")
except (concurrent.futures.TimeoutError, TimeoutError):
    elapsed = time.time() - start
    test(f"Timeout enforcement: caught in {elapsed:.1f}s",
         elapsed < 2.0,
         f"took {elapsed:.1f}s")

# Test _call_with_timeout allows fast backends
def fast_backend(prompt, system_prompt):
    return "fast", True, {"latency_s": 0.01}

result = orch._call_with_timeout(fast_backend, "test", "", timeout_s=5)
test("Fast backend passes through", result == ("fast", True, {"latency_s": 0.01}))

# ═══════════════════════════════════════════════════════════════
# TEST 4: TOKEN ESTIMATION CONSISTENCY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: TOKEN ESTIMATION — Consistenza tra moduli")
print("=" * 70)

from retrieval.context_builder import ContextBuilder

cb = ContextBuilder.__new__(ContextBuilder)
# Both should use same ratio
sample_text = "A" * 4000  # 4000 chars

orch_tokens = orch._estimate_tokens(sample_text)
cb_tokens = cb._count_tokens(sample_text)

test(f"Orchestrator stima: {orch_tokens} tokens per 4000 chars",
     900 <= orch_tokens <= 1100, f"expected ~1000, got {orch_tokens}")
test(f"ContextBuilder stima: {cb_tokens} tokens per 4000 chars",
     900 <= cb_tokens <= 1100, f"expected ~1000, got {cb_tokens}")
test("Stime coerenti (differenza < 25%)",
     abs(orch_tokens - cb_tokens) / max(orch_tokens, cb_tokens) < 0.25,
     f"orch={orch_tokens}, cb={cb_tokens}")

# ═══════════════════════════════════════════════════════════════
# TEST 5: REASONING PROMPT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: REASONING PROMPT — Istruzioni per ragionamento")
print("=" * 70)

from config.prompts import get_system_prompt

prompt = get_system_prompt("generale")

test("Prompt contiene 'RAGIONAMENTO STRUTTURATO'",
     "RAGIONAMENTO STRUTTURATO" in prompt)
test("Prompt contiene 'COMPRENDI'",
     "COMPRENDI" in prompt)
test("Prompt contiene 'ANALIZZA'",
     "ANALIZZA" in prompt)
test("Prompt contiene 'RAGIONA'",
     "RAGIONA" in prompt)
test("Prompt contiene 'SINTETIZZA'",
     "SINTETIZZA" in prompt)
test("Prompt contiene 'VERIFICA'",
     "VERIFICA" in prompt)
test("Prompt contiene 'non solo i dati'",
     "non solo i dati" in prompt)
test("Prompt contiene istruzioni anti-allucinazione",
     "ANTI-ALLUCINAZIONE" in prompt)

# ═══════════════════════════════════════════════════════════════
# RIEPILOGO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"RISULTATO: {PASS}/{total} test passati")
if FAIL:
    print(f"⚠️  {FAIL} test FALLITI")
else:
    print("🎉 TUTTI I TEST PASSATI!")
print("=" * 70)

sys.exit(0 if FAIL == 0 else 1)
