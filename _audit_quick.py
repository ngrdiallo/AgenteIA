"""Quick audit of current system state — facts only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.query_classifier import classify_query

tests = [
    ("ciao", "greeting"),
    ("quante pagine ha?", "meta"),
    ("chi era Bernini?", "focused"),
    ("analizza il file", "comprehensive"),
    ("parlami del Barocco", "focused"),
    ("cos'è il Manierismo?", "focused"),
    ("che differenza c'è tra Bernini e Borromini?", "focused"),
    ("potresti analizzare il file?", "comprehensive"),
    ("ma", "greeting"),  # short no-info query
    ("e Bernini?", "focused"),  # context-dependent follow-up
    ("ciao, dimmi tutto sul documento", "comprehensive"),
    ("il file quante pagine ha?", "meta"),
    # Edge cases: typos, dialect, indirect
    ("analisami il file", "comprehensive"),  # typo: analisami
    ("ch'è il barocco?", "focused"),  # informal contraction
    ("me lo riassumi?", "comprehensive"),  # pronome enclitico
]

fail = 0
for q, exp in tests:
    r = classify_query(q)
    ok = r.query_type == exp
    if not ok:
        fail += 1
    status = "OK" if ok else "FAIL"
    print(f"[{status}] '{q}' -> {r.query_type} (expected {exp}, conf={r.confidence})")

print(f"\n{len(tests) - fail}/{len(tests)} correct, {fail} failures")
