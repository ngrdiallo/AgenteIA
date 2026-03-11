#!/usr/bin/env python3
"""
Test E2E con domande stile utente reale italiano.
Verifica che l'IA risponda con ragionamento, non pattern hardcoded.
Testa anche il timeout enforcement (nessun backend >60s).

Esegui con server attivo su localhost:8001:
  python test_user_style_e2e.py
"""

import requests
import json
import sys
import time

BASE = "http://localhost:8001"
TIMEOUT = 70  # 70s max per request (backend timeout 55s + overhead)
RESULTS = []


def query_sync(text: str, modalita: str = "generale") -> dict:
    """Invia query al server e ritorna la risposta."""
    try:
        r = requests.post(
            f"{BASE}/api/chat/sync",
            json={"query": text, "modalita": modalita, "use_rag": True},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ReadTimeout:
        return {"answer": "", "backend_used": "TIMEOUT", "latency_s": TIMEOUT, "model": ""}
    except Exception as e:
        return {"answer": "", "backend_used": "ERROR", "latency_s": 0, "model": "", "error": str(e)}


def evaluate(name: str, query: str, result: dict, checks: dict):
    """Valuta una risposta con criteri specifici."""
    answer = result.get("answer", "")
    latency = result.get("latency_s", 0)
    backend = result.get("backend_used", "?")
    model = result.get("model", "?")

    passed = 0
    total = 0
    details = []

    # Check lunghezza minima
    if "min_len" in checks:
        total += 1
        if len(answer) >= checks["min_len"]:
            passed += 1
        else:
            details.append(f"troppo corta: {len(answer)} < {checks['min_len']}")

    # Check keyword presenti
    if "has_keywords" in checks:
        for kw in checks["has_keywords"]:
            total += 1
            if kw.lower() in answer.lower():
                passed += 1
            else:
                details.append(f"keyword mancante: '{kw}'")

    # Check NON contiene
    if "not_contains" in checks:
        for nc in checks["not_contains"]:
            total += 1
            if nc.lower() not in answer.lower():
                passed += 1
            else:
                details.append(f"contiene erroneamente: '{nc}'")

    # Check latenza massima (timeout enforcement)
    if "max_latency" in checks:
        total += 1
        if latency <= checks["max_latency"]:
            passed += 1
        else:
            details.append(f"latenza {latency:.1f}s > max {checks['max_latency']}s")

    # Check non errore
    total += 1
    if backend not in ("TIMEOUT", "ERROR", "none"):
        passed += 1
    else:
        details.append(f"backend error: {backend}")

    score = round(passed / total * 10, 1) if total else 0
    status = "✅" if score >= 8.0 else "⚠️" if score >= 5 else "❌"

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: \"{query}\"")
    print(f"-"*60)
    print(f"Backend: {backend} ({model})")
    print(f"Latenza: {latency:.1f}s")
    print(f"Risposta ({len(answer)} char): {answer[:300]}...")
    print(f"\nSCORE: {score}/10")
    if details:
        for d in details:
            print(f"  ⚠️ {d}")
    else:
        print(f"  {status} Tutti i criteri superati!")

    RESULTS.append({
        "name": name,
        "score": score,
        "latency": latency,
        "backend": backend,
        "model": model,
        "answer_len": len(answer),
        "issues": details,
    })
    return score


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════
print("🔍 Health check...")
try:
    h = requests.get(f"{BASE}/api/health", timeout=5).json()
    docs = h.get("documents_indexed", 0)
    print(f"Server: {h.get('status', '?')} | Documenti: {docs}")
    if docs == 0:
        print("⚠️ Nessun documento indicizzato. I test RAG falliranno.")
except Exception as e:
    print(f"❌ Server non raggiungibile: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# TEST SUITE: DOMANDE STILE UTENTE REALE
# ═══════════════════════════════════════════════════════════════

# --- FORME CORTESI (FIX AUDIT: conditional verbs) ---

r = query_sync("potresti analizzare il file per favore?")
evaluate(
    "CORTESE: potresti analizzare",
    "potresti analizzare il file per favore?",
    r,
    {"min_len": 500, "has_keywords": ["pagina", "arte"], "max_latency": 60},
)

r = query_sync("sarebbe possibile fare un riassunto del documento?")
evaluate(
    "CORTESE: sarebbe possibile riassunto",
    "sarebbe possibile fare un riassunto del documento?",
    r,
    {"min_len": 500, "has_keywords": ["riassunto"], "max_latency": 60},
)

# --- DOMANDE NATURALI (stile studente) ---

r = query_sync("mi parli del Barocco? vorrei capire le caratteristiche principali")
evaluate(
    "NATURALE: parlami del Barocco",
    "mi parli del Barocco? vorrei capire le caratteristiche principali",
    r,
    {"min_len": 200, "has_keywords": ["barocco"], "max_latency": 60},
)

r = query_sync("che differenza c'è tra Bernini e Borromini?")
evaluate(
    "RAGIONAMENTO: confronto Bernini vs Borromini",
    "che differenza c'è tra Bernini e Borromini?",
    r,
    {"min_len": 200, "has_keywords": ["bernini", "borromini"], "max_latency": 60},
)

r = query_sync("fammi un riassunto veloce del documento, tipo 10 righe")
evaluate(
    "CORTESE: fammi un riassunto veloce",
    "fammi un riassunto veloce del documento, tipo 10 righe",
    r,
    {"min_len": 200, "max_latency": 60},
)

# --- TIMEOUT ENFORCEMENT: nessun backend deve superare 60s ---

r = query_sync("cos'è il Manierismo?")
evaluate(
    "TIMEOUT: cos'è il Manierismo (max 60s)",
    "cos'è il Manierismo?",
    r,
    {"min_len": 100, "has_keywords": ["manierismo"], "max_latency": 60},
)

# --- META e GREETING ---

r = query_sync("quante pagine ha il pdf sulla storia dell'arte?")
evaluate(
    "META: quante pagine ha il pdf",
    "quante pagine ha il pdf sulla storia dell'arte?",
    r,
    {"min_len": 50, "has_keywords": ["pagine"], "max_latency": 15},
)

r = query_sync("ciao! cosa puoi fare?")
evaluate(
    "GREETING: ciao cosa puoi fare",
    "ciao! cosa puoi fare?",
    r,
    {"min_len": 50, "max_latency": 15},
)

# --- NEGAZIONE (fix audit) ---

r = query_sync("non voglio un riassunto, dimmi solo gli artisti principali del documento")
evaluate(
    "NEGAZIONE: non riassunto, solo artisti",
    "non voglio un riassunto, dimmi solo gli artisti principali del documento",
    r,
    {"min_len": 100, "not_contains": ["ecco il riassunto"], "max_latency": 60},
)


# ═══════════════════════════════════════════════════════════════
# RIEPILOGO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("REPORT FINALE — DOMANDE STILE UTENTE")
print("=" * 60)
total_score = 0
for r in RESULTS:
    status = "✅" if r["score"] >= 8 else "⚠️" if r["score"] >= 5 else "❌"
    print(f"  {status} {r['name']:<50} {r['score']:>5}/10  [{r['backend']}/{r['model']}] {r['latency']:.1f}s")
    total_score += r["score"]

avg = total_score / len(RESULTS) if RESULTS else 0
print(f"\nMEDIA TOTALE: {avg:.1f}/10")
print(f"Test superati (≥8): {sum(1 for r in RESULTS if r['score'] >= 8)}/{len(RESULTS)}")

# Check timeout enforcement globale
max_lat = max(r["latency"] for r in RESULTS)
if max_lat > 60:
    print(f"\n⚠️ TIMEOUT ENFORCEMENT FALLITO: latenza max = {max_lat:.1f}s (>60s)")
    print("  → Il fix dell'orchestrator non funziona correttamente")
else:
    print(f"\n✅ TIMEOUT ENFORCEMENT OK: latenza max = {max_lat:.1f}s (≤60s)")

sys.exit(0 if avg >= 7.0 else 1)
