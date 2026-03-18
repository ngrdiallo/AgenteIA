"""Test COMPLETO del classificatore con tutte le query problematiche reali."""
import sys
sys.path.insert(0, ".")
from retrieval.query_classifier import classify_query

# Ogni test: (query, tipo_atteso, descrizione)
tests = [
    # === ERRORI STORICI (dal chat log reale) ===
    ("ciao analizza il file, quante pagine ha? fai un riassunto del doc", "comprehensive", "MSG1: la query che falliva 5 volte"),
    ("devi analizzare il file. quante pagine ha. devi farne un riassunto.", "comprehensive", "MSG13: classificata come META erroneamente"),
    ("Analizza l'intero documento Storia dell'Arte Moderna", "comprehensive", "MSG10: questa funzionava"),
    ("storia dell'arte moderna 13 o", "focused", "MSG6: nome file vago"),

    # === VERBI ITALIANI CONIUGATI ===
    ("analizzare il file pdf", "comprehensive", "Infinito: analizzare"),
    ("analizzami questo documento", "comprehensive", "Imperativo+pronome: analizzami"),
    ("devi analizzarlo tutto", "comprehensive", "devi + pronome: analizzarlo"),
    ("voglio un riassunto del documento", "comprehensive", "voglio riassunto"),
    ("vorrei una sintesi completa", "comprehensive", "vorrei sintesi"),
    ("puoi riassumere il contenuto?", "comprehensive", "puoi riassumere"),
    ("riassumimi il file", "comprehensive", "riassumimi"),
    ("descrivimi tutto il contenuto", "comprehensive", "descrivimi tutto"),
    ("spiegami cosa contiene il documento", "comprehensive", "spiegami + cosa contiene"),

    # === META PURE ===
    ("quante pagine ha?", "meta", "META puro: pagine"),
    ("che file ho caricato?", "meta", "META puro: file caricati"),
    ("quanti chunk ha il documento?", "meta", "META puro: chunk"),

    # === FOCUSED ===
    ("chi era Bernini?", "focused", "Focused: chi era"),
    ("cos'è il Barocco?", "focused", "Focused: cos'è"),
    ("come funziona il Barocco?", "focused", "Focused: come funziona (non greeting)"),
    ("differenza tra Bernini e Borromini", "focused", "Focused: differenza"),
    ("parlami del Rinascimento", "focused", "Focused: parlami di"),

    # === GREETING ===
    ("ciao", "greeting", "Greeting puro"),
    ("ciao come stai?", "greeting", "Greeting: ciao come stai"),
    ("buongiorno", "greeting", "Greeting: buongiorno"),
    ("buonasera come va?", "greeting", "Greeting composto"),

    # === COMPREHENSIVE CON SALUTO ===
    ("buonasera, dimmi tutto sul Barocco", "comprehensive", "Saluto + comprehensive"),
    ("ciao, fai un riassunto del documento", "comprehensive", "Saluto + riassunto"),
    ("salve, analizza il file per favore", "comprehensive", "Saluto + analizza"),
]

passed = 0
failed = 0
for query, expected, desc in tests:
    r = classify_query(query)
    ok = r.query_type == expected
    if ok:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    print(f"[{status}] {desc}")
    if not ok:
        print(f"       Query: \"{query}\"")
        print(f"       Atteso: {expected}, Ottenuto: {r.query_type}")
        print(f"       Intents: {r.detected_intents}")
    
print(f"\n{'='*50}")
print(f"RISULTATO: {passed}/{len(tests)} superati, {failed} falliti")
if failed == 0:
    print("TUTTI I TEST SUPERATI!")
else:
    print(f"ATTENZIONE: {failed} test da correggere")
