"""Quick speed test for meta + greeting."""
import requests, time

BASE = "http://localhost:8001"

# Warm up
print("Warming up...")
r = requests.post(f"{BASE}/api/chat/sync", json={"query": "warm up"}, timeout=60)
print(f"Warmup done: {r.status_code}")

tests = [
    ("quante pagine ha il documento?", "meta", 5),
    ("che tipo di file è?", "meta", 5),
    ("quali documenti ho caricato?", "meta", 5),
    ("ciao", "greeting", 5),
    ("chi è Caravaggio?", "focused", 20),
]

for query, qtype, limit in tests:
    t = time.time()
    r = requests.post(f"{BASE}/api/chat/sync", json={"query": query}, timeout=limit + 10)
    elapsed = time.time() - t
    data = r.json()
    backend = data.get("backend_used", "?")
    answer = data.get("answer", "")[:80].replace("\n", " ")
    status = "OK" if elapsed <= limit else "SLOW"
    print(f"  {status} [{elapsed:.1f}s/{backend}] {qtype}: \"{query}\" → {answer}...")
