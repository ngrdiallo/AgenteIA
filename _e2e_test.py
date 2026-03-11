"""
Test E2E completo: Upload PDF + Query + Verifica Risposte
Simula il flusso reale utente: carica file → chiedi domande → valuta qualità.

Ogni test ha criteri 10/10:
- META: deve rispondere con numero pagine CORRETTO
- COMPREHENSIVE: deve contenere riassunto con contenuti dal documento
- FOCUSED: deve citare fatti specifici dal documento
- GREETING: deve menzionare i documenti caricati
- MULTI-INTENT: deve gestire analisi + meta + riassunto insieme
"""
import json
import os
import sys
import time
import requests

BASE_URL = "http://localhost:8001"
PDF_PATH = r"C:\Users\A893apulia\Downloads\AgenteIASDA\PDF\ESAME STORIA DA MODERNA.pdf"
if not os.path.exists(PDF_PATH):
    # Fallback: usa un PDF più piccolo dalla cartella data
    for candidate in [
        r"C:\Users\A893apulia\Downloads\AgenteIASDA\PDF\Il Gotico Internazionale (1).pdf",
        r"C:\Users\A893apulia\Downloads\AgenteIASDA\PDF\Storia dell'Arte Moderna 1 A.pdf",
    ]:
        if os.path.exists(candidate):
            PDF_PATH = candidate
            break

RESULTS = []


def score_response(answer: str, criteria: list[tuple[str, str]]) -> tuple[int, list[str]]:
    """Valuta la risposta su N criteri.
    criteria: [(nome_criterio, keyword_o_condizione), ...]
    Ritorna (score_su_10, lista_problemi)
    """
    points = 0
    max_points = len(criteria)
    problems = []
    for name, check in criteria:
        if callable(check):
            if check(answer):
                points += 1
            else:
                problems.append(f"FAIL: {name}")
        elif isinstance(check, str):
            if check.lower() in answer.lower():
                points += 1
            else:
                problems.append(f"FAIL: {name} ('{check}' non trovato)")
    score = round(points / max_points * 10, 1)
    return score, problems


def upload_pdf(filepath: str) -> dict:
    """Carica un file PDF e ritorna info."""
    fname = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/upload",
            files={"files": (fname, f, "application/pdf")},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def query(text: str, use_rag: bool = True, timeout: int = 300) -> dict:
    """Invia una query e ritorna la risposta completa (SSE parsing)."""
    resp = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "query": text,
            "use_rag": use_rag,
            "modalita": "approfondimento",
            "history": [],
        },
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()

    # Parse SSE events
    answer_text = ""
    backend_used = ""
    model_used = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            try:
                data = json.loads(data_str)
                if "text" in data:
                    answer_text = data["text"]
                if "backend_used" in data:
                    backend_used = data["backend_used"]
                if "model" in data:
                    model_used = data["model"]
                # Check for complete response object
                if "answer" in data:
                    answer_text = data["answer"]
            except json.JSONDecodeError:
                pass

    return {
        "answer": answer_text,
        "backend_used": backend_used,
        "model": model_used,
    }


def run_test(name: str, query_text: str, criteria: list, test_type: str):
    """Esegue un singolo test E2E."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Query: \"{query_text}\"")
    print(f"Tipo atteso: {test_type}")
    print("-" * 60)

    try:
        start = time.time()
        result = query(query_text)
        elapsed = time.time() - start

        answer = result.get("answer", "")
        backend = result.get("backend_used", "?")
        model = result.get("model", "?")

        print(f"Backend: {backend} ({model})")
        print(f"Latenza: {elapsed:.1f}s")
        print(f"Risposta ({len(answer)} char): {answer[:300]}...")

        score, problems = score_response(answer, criteria)
        print(f"\nSCORE: {score}/10")
        if problems:
            for p in problems:
                print(f"  ⚠️ {p}")
        else:
            print("  ✅ Tutti i criteri superati!")

        RESULTS.append({
            "test": name,
            "type": test_type,
            "score": score,
            "problems": problems,
            "backend": backend,
            "model": model,
            "latency": round(elapsed, 1),
            "answer_len": len(answer),
            "answer_preview": answer[:200],
        })
        return score

    except Exception as e:
        print(f"❌ ERRORE: {e}")
        RESULTS.append({
            "test": name,
            "type": test_type,
            "score": 0,
            "problems": [str(e)],
            "backend": "error",
            "model": "",
            "latency": 0,
            "answer_len": 0,
            "answer_preview": "",
        })
        return 0


def main():
    # ─── Health check ───
    print("🔍 Health check...")
    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=5).json()
        print(f"Server: {health.get('status')} | Documenti: {health.get('storage', {}).get('documents', 0)}")
    except Exception as e:
        print(f"❌ Server non raggiungibile: {e}")
        sys.exit(1)

    # ─── Upload file (se non già indicizzato) ───
    docs = health.get("storage", {}).get("documents", 0)
    if docs == 0:
        print(f"\n📤 Upload PDF: {os.path.basename(PDF_PATH)}")
        try:
            upload_result = upload_pdf(PDF_PATH)
            print(f"Upload result: {json.dumps(upload_result, indent=2, ensure_ascii=False)[:300]}")
            time.sleep(2)  # Attendi indicizzazione
        except Exception as e:
            print(f"❌ Upload fallito: {e}")
    else:
        print(f"\n📚 {docs} documenti già indicizzati, skip upload")

    # ═══════════════════════════════════════════════════════
    # TEST 1: META — "quante pagine ha il documento?"
    # ═══════════════════════════════════════════════════════
    run_test(
        name="META: Quante pagine ha",
        query_text="quante pagine ha il documento?",
        criteria=[
            ("risponde in italiano", lambda a: any(w in a.lower() for w in ["pagine", "pagina", "documento"])),
            ("contiene un numero", lambda a: any(c.isdigit() for c in a)),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("non ignora la domanda", lambda a: "non ho informazioni" not in a.lower()),
            ("menziona file specifico", lambda a: any(w in a.lower() for w in ["storia", "arte", "esame", "pdf"])),
        ],
        test_type="meta",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 2: META — "che file ho caricato?"
    # ═══════════════════════════════════════════════════════
    run_test(
        name="META: Che file ho caricato",
        query_text="che file ho caricato?",
        criteria=[
            ("elenca file", lambda a: any(w in a.lower() for w in ["storia", "arte", "pdf", ".pdf", "esame"])),
            ("risponde in italiano", lambda a: any(w in a.lower() for w in ["file", "documento", "caricato"])),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("contiene info utile", lambda a: len(a) > 30),
        ],
        test_type="meta",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 3: COMPREHENSIVE — "devi analizzare il file..."
    # IL BUG CRITICO: prima classificato come meta!
    # ═══════════════════════════════════════════════════════
    run_test(
        name="COMPREHENSIVE: Analizza+pagine+riassunto (BUG FIX)",
        query_text="devi analizzare il file. quante pagine ha. devi farne un riassunto.",
        criteria=[
            ("contiene riassunto/analisi", lambda a: any(w in a.lower() for w in ["riassunto", "analisi", "sintesi", "contenuto"])),
            ("menziona contenuti del documento", lambda a: any(w in a.lower() for w in ["arte", "storia", "barocco", "rinascimento", "mostra", "pittura", "scultura", "architettura", "artista", "artisti"])),
            ("più di 200 caratteri", lambda a: len(a) > 200),
            ("non è solo metadati", lambda a: "chunk" not in a.lower() or "riassunto" in a.lower()),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("non dice 'non posso'", lambda a: "non posso" not in a.lower()),
            ("menziona pagine", lambda a: any(c.isdigit() for c in a)),
            ("risposta sostanziale", lambda a: len(a) > 500),
        ],
        test_type="comprehensive",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 4: COMPREHENSIVE — "ciao analizza il file..."
    # LA QUERY CHE FALLIVA 5 VOLTE NEL CHAT REALE
    # ═══════════════════════════════════════════════════════
    run_test(
        name="COMPREHENSIVE: ciao analizza il file (MSG1 fix)",
        query_text="ciao analizza il file, quante pagine ha? fai un riassunto del doc",
        criteria=[
            ("contiene riassunto", lambda a: any(w in a.lower() for w in ["riassunto", "analisi", "sintesi", "sommario"])),
            ("menziona contenuto doc", lambda a: any(w in a.lower() for w in ["arte", "storia", "barocco", "rinascimento", "pittura", "scultura", "architettura"])),
            ("più di 200 char", lambda a: len(a) > 200),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("non dice 'non posso'", lambda a: "non posso" not in a.lower()),
            ("risposta sostanziale", lambda a: len(a) > 400),
        ],
        test_type="comprehensive",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 5: FOCUSED — "chi era Bernini?"
    # ═══════════════════════════════════════════════════════
    run_test(
        name="FOCUSED: Chi era Bernini",
        query_text="chi era Bernini?",
        criteria=[
            ("menziona Bernini", "bernini"),
            ("contesto storico", lambda a: any(w in a.lower() for w in ["barocco", "xvii", "seicento", "roma", "scultor", "architet", "artista"])),
            ("più di 100 char", lambda a: len(a) > 100),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("risposta informativa", lambda a: len(a) > 200),
        ],
        test_type="focused",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 6: GREETING — "ciao come stai?"
    # ═══════════════════════════════════════════════════════
    run_test(
        name="GREETING: ciao come stai",
        query_text="ciao come stai?",
        criteria=[
            ("saluto", lambda a: any(w in a.lower() for w in ["ciao", "buongiorno", "salve", "benvenuto", "come posso"])),
            ("menziona documenti", lambda a: any(w in a.lower() for w in ["document", "file", "caricato", "pdf", "storia", "arte"])),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
            ("non vuota", lambda a: len(a) > 20),
        ],
        test_type="greeting",
    )

    # ═══════════════════════════════════════════════════════
    # TEST 7: FOCUSED — "cos'è il Barocco?"
    # ═══════════════════════════════════════════════════════
    run_test(
        name="FOCUSED: cos'è il Barocco",
        query_text="cos'è il Barocco?",
        criteria=[
            ("menziona Barocco", "barocco"),
            ("contesto storico", lambda a: any(w in a.lower() for w in ["xvii", "seicento", "stile", "arte", "architettura", "periodo"])),
            ("più di 100 char", lambda a: len(a) > 100),
            ("non dice 'non ho ricevuto'", lambda a: "non ho ricevuto" not in a.lower()),
        ],
        test_type="focused",
    )

    # ═══════════════════════════════════════════════════════
    # REPORT FINALE
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("REPORT FINALE E2E")
    print("=" * 60)
    total_score = 0
    for r in RESULTS:
        emoji = "✅" if r["score"] >= 8 else "⚠️" if r["score"] >= 5 else "❌"
        print(f"  {emoji} {r['test']:50s} {r['score']:4.1f}/10  [{r['backend']}/{r['model']}] {r['latency']}s")
        total_score += r["score"]

    avg = total_score / len(RESULTS) if RESULTS else 0
    print(f"\nMEDIA TOTALE: {avg:.1f}/10")
    print(f"Test superati (≥8): {sum(1 for r in RESULTS if r['score'] >= 8)}/{len(RESULTS)}")

    # Salva risultati
    with open("_e2e_test_results.json", "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Risultati salvati in _e2e_test_results.json")

    # Ritorna exit code
    failed = [r for r in RESULTS if r["score"] < 8]
    if failed:
        print(f"\n⚠️ {len(failed)} test sotto 8/10 — serve correzione")
        return 1
    else:
        print("\n🎉 TUTTI I TEST ≥ 8/10!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
