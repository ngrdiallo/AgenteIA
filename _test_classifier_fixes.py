"""Quick test: verify classifier fixes and pattern count."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from retrieval.query_classifier import (
    _COMPREHENSIVE_SIGNALS, _META_SIGNALS, _FOCUSED_SIGNALS,
    _GREETING_PATTERNS, _GREETING_PHRASES, classify_query
)

comp = len(_COMPREHENSIVE_SIGNALS)
meta = len(_META_SIGNALS)
foc = len(_FOCUSED_SIGNALS)
greet = len(_GREETING_PATTERNS)
phrases = len(_GREETING_PHRASES)
total = comp + meta + foc + greet + phrases
print(f"Comprehensive: {comp}")
print(f"Meta: {meta}")
print(f"Focused: {foc}")
print(f"Greeting patterns: {greet}")
print(f"Greeting phrases: {phrases}")
print(f"TOTAL: {total}")
print(f"UNDER 60: {total <= 60}")
print()

tests = [
    ("e Bernini?", "focused"),
    ("analisami il file", "comprehensive"),
    ("ciao", "greeting"),
    ("quante pagine ha il documento?", "meta"),
    ("parlami del Barocco", "focused"),
    ("analizza tutto il contenuto", "comprehensive"),
    ("buongiorno", "greeting"),
    ("chi è Caravaggio?", "focused"),
    ("fammi un riassunto completo", "comprehensive"),
    ("quanto è lungo il pdf?", "meta"),
    ("ciao, analizza il file", "comprehensive"),
    ("Michelangelo?", "focused"),
    ("grazie mille", "greeting"),
    ("cosa contiene il documento?", "comprehensive"),
    ("Raffaello", "focused"),
    ("puoi analizzare il documento?", "comprehensive"),
    ("potresti riassumere tutto?", "comprehensive"),
    ("analizzami questo pdf", "comprehensive"),
    ("dimmi tutto sul contenuto", "comprehensive"),
    ("che tipo di file è?", "meta"),
]

ok = 0
for q, expected in tests:
    result = classify_query(q)
    qt = result.query_type
    status = "OK" if qt == expected else "FAIL"
    if qt == expected:
        ok += 1
    print(f"  {status}: \"{q}\" -> {qt} (expected: {expected})")

print(f"\n{ok}/{len(tests)} passed")
