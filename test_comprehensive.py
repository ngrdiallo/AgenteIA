"""
Test: verifica che l'analisi comprensiva copra tutto il documento.
"""

import sys
sys.path.insert(0, r"C:\Users\A893apulia\Downloads\AgenteIASDA\IAGestioneArte")

import requests
import json

BASE = "http://localhost:8001"

def test_query_classifier():
    """Test classificatore query."""
    from retrieval.query_classifier import classify_query

    # Comprehensive queries
    tests_comp = [
        "analizza il file. parlami del contenuto e di tutte le informazioni che riesci a ricavare in proposito.",
        "riassumi tutto il documento",
        "dimmi tutto quello che contiene il file",
        "parlami di tutti gli argomenti trattati",
        "analizza il documento completamente",
        "di cosa parla il file?",
        "quali sono i temi principali del documento?",
    ]
    print("=== COMPREHENSIVE QUERIES ===")
    for q in tests_comp:
        c = classify_query(q)
        status = "✅" if c.query_type == "comprehensive" else "❌"
        print(f"  {status} [{c.query_type}] conf={c.confidence:.2f} top_k={c.recommended_top_k} → {q[:60]}...")

    # Focused queries
    tests_focused = [
        "cosa dice il documento sulla prospettiva?",
        "chi era Brunelleschi?",
        "parla di Masaccio",
    ]
    print("\n=== FOCUSED QUERIES ===")
    for q in tests_focused:
        c = classify_query(q)
        status = "✅" if c.query_type == "focused" else "❌"
        print(f"  {status} [{c.query_type}] conf={c.confidence:.2f} top_k={c.recommended_top_k} → {q}")

    # Greeting
    tests_greet = ["ciao chi sei", "buongiorno"]
    print("\n=== GREETING QUERIES ===")
    for q in tests_greet:
        c = classify_query(q)
        status = "✅" if c.query_type == "greeting" else "❌"
        print(f"  {status} [{c.query_type}] conf={c.confidence:.2f} → {q}")


def test_comprehensive_chat():
    """Test che l'endpoint chat usi il path comprensivo."""
    print("\n=== COMPREHENSIVE CHAT ENDPOINT ===")

    payload = {
        "query": "analizza il file. parlami del contenuto e di tutte le informazioni che riesci a ricavare in proposito.",
        "use_rag": True,
        "modalita": "storico_artistico",
    }

    print(f"  Sending: {payload['query'][:60]}...")
    r = requests.post(f"{BASE}/api/chat/sync", json=payload, timeout=120)
    print(f"  Status: {r.status_code}")

    if r.status_code == 200:
        data = r.json()
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        word_count = len(answer.split())
        pages_cited = set()
        for c in citations:
            pages_cited.add(c.get("page_number", 0))

        print(f"  Parole risposta: {word_count}")
        print(f"  Citazioni: {len(citations)}")
        print(f"  Pagine citate: {sorted(pages_cited)}")
        print(f"  Backend: {data.get('backend_used', '?')}")

        # Verifica copertura
        if len(pages_cited) > 10:
            print(f"  ✅ Copertura pagine ampia: {len(pages_cited)} pagine")
        else:
            print(f"  ⚠️  Copertura pagine limitata: solo {len(pages_cited)} pagine")

        # Verifica lunghezza
        if word_count > 300:
            print(f"  ✅ Risposta sufficientemente lunga: {word_count} parole")
        else:
            print(f"  ⚠️  Risposta troppo breve: {word_count} parole")

        # Mostra primi 500 caratteri
        print(f"\n  === INIZIO RISPOSTA ===")
        print(f"  {answer[:500]}...")
        print(f"  === FINE (troncata) ===")
    else:
        print(f"  ❌ Errore: {r.text[:300]}")


if __name__ == "__main__":
    test_query_classifier()
    test_comprehensive_chat()
    print("\n✅ Test completati")
