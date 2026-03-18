"""Test per verificare il filtro di rilevanza delle citazioni."""
import requests
import json

BASE = "http://localhost:8001"

def test_query(label, query):
    print(f"\n=== {label} ===")
    print(f"Query: {query}")
    r = requests.post(f"{BASE}/api/chat", json={
        "query": query, "modalita": "esperto", "use_rag": True, "history": []
    }, stream=True)
    
    citations = []
    answer = ""
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
                if "citations" in d:
                    citations = d["citations"]
                if "text" in d:
                    answer = d["text"]
            except:
                pass
    
    print(f"  Citazioni: {len(citations)}")
    for c in citations:
        score = c.get("score", 0)
        print(f"    - {c['source_file']} p.{c['page_number']} (score={score:.3f})")
    print(f"  Risposta: {answer[:200]}")
    return len(citations)

# Test 1: Saluto — NON deve avere citazioni
n1 = test_query("TEST 1: Saluto generico", "ciao chi sei devi analizzare file pdf")

# Test 2: Contenuto — DEVE avere citazioni
n2 = test_query("TEST 2: Domanda sul contenuto", "Cosa dice il documento su Brunelleschi?")

# Test 3: Fuori tema — NON deve avere citazioni
n3 = test_query("TEST 3: Fuori tema", "Come si programma in Python?")

# Test 4: Contenuto specifico — DEVE avere citazioni
n4 = test_query("TEST 4: Prospettiva rinascimentale", "Cos'è la prospettiva rinascimentale?")

print("\n" + "=" * 60)
print("RISULTATI:")
print(f"  Test 1 (saluto): {n1} citazioni {'✅ CORRETTO' if n1 == 0 else '❌ ERRORE'}")
print(f"  Test 2 (contenuto): {n2} citazioni {'✅ CORRETTO' if n2 > 0 else '❌ ERRORE'}")
print(f"  Test 3 (fuori tema): {n3} citazioni {'✅ CORRETTO' if n3 == 0 else '❌ ERRORE'}")
print(f"  Test 4 (contenuto): {n4} citazioni {'✅ CORRETTO' if n4 > 0 else '❌ ERRORE'}")
