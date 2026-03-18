"""Test completo di tutti gli endpoint HTTP live."""
import requests
import sys

BASE = "http://localhost:8001"
ok = 0
fail = 0


def test(name, fn):
    global ok, fail
    try:
        fn()
        print(f"  [PASS] {name}")
        ok += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        fail += 1


# ── UI ──────────────────────────────────────────────────────
print("=" * 60)
print("TEST ENDPOINT HTTP COMPLETI")
print("=" * 60)

print("\n--- UI ---")


def t_ui():
    r = requests.get(f"{BASE}/")
    assert r.status_code == 200, f"status {r.status_code}"
    assert "<html" in r.text.lower(), "No HTML tag found"


test("GET / (UI HTML)", t_ui)


# ── Health ──────────────────────────────────────────────────
print("\n--- Health ---")


def t_health():
    r = requests.get(f"{BASE}/api/health")
    assert r.status_code == 200
    d = r.json()
    assert "status" in d
    assert "backends" in d
    assert "storage" in d
    st = d["status"]
    docs = d["storage"]["documents"]
    print(f"    status={st}, docs={docs}")


test("GET /api/health", t_health)


# ── Modalita ────────────────────────────────────────────────
print("\n--- Modalita ---")


def t_modalita():
    r = requests.get(f"{BASE}/api/modalita")
    assert r.status_code == 200
    d = r.json()
    assert len(d["modalita"]) == 8, f"Expected 8, got {len(d['modalita'])}"
    nomi = [m["id"] for m in d["modalita"]]
    print(f"    {nomi}")


test("GET /api/modalita (8)", t_modalita)


# ── Documents (empty) ──────────────────────────────────────
print("\n--- Documents ---")


def t_docs_empty():
    r = requests.get(f"{BASE}/api/documents")
    assert r.status_code == 200
    d = r.json()
    print(f"    total={d['total']}")


test("GET /api/documents", t_docs_empty)


# ── Upload ──────────────────────────────────────────────────
print("\n--- Upload ---")


def t_upload_invalid():
    r = requests.post(
        f"{BASE}/api/documents/upload",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"


test("POST upload formato invalido (400)", t_upload_invalid)


def t_upload_pdf():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(
        200, 10,
        text="Il Rinascimento italiano: Leonardo, Michelangelo, Raffaello",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.cell(
        200, 10,
        text="La Gioconda fu dipinta da Leonardo da Vinci nel XVI secolo.",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.cell(
        200, 10,
        text="Il David di Michelangelo si trova nella Galleria Accademia di Firenze.",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf_bytes = pdf.output()
    r = requests.post(
        f"{BASE}/api/documents/upload",
        files={"file": ("arte_test.pdf", pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200, f"Upload failed: {r.status_code} {r.text}"
    d = r.json()
    assert d["chunks"] > 0
    print(f"    filename={d['filename']}, pages={d['pages']}, chunks={d['chunks']}")


test("POST upload PDF valido (200)", t_upload_pdf)


def t_docs_after():
    r = requests.get(f"{BASE}/api/documents")
    assert r.status_code == 200
    d = r.json()
    assert d["total"] >= 1, f"Expected >=1, got {d['total']}"
    fnames = [doc["filename"] for doc in d["documents"]]
    print(f"    total={d['total']}, files={fnames}")


test("GET /api/documents dopo upload", t_docs_after)


# ── Chat SSE ───────────────────────────────────────────────
print("\n--- Chat ---")


def t_chat_rag():
    r = requests.post(
        f"{BASE}/api/chat",
        json={
            "query": "Parlami di Leonardo da Vinci",
            "modalita": "generale",
            "history": [],
            "stream": True,
        },
        timeout=30,
    )
    assert r.status_code == 200, f"status {r.status_code}"
    ct = r.headers.get("content-type", "")
    assert "text/event-stream" in ct, f"Wrong CT: {ct}"
    text = r.text
    assert "event: thinking" in text, "Missing 'thinking' event"
    assert "event: done" in text, "Missing 'done' event"
    if "arte_test.pdf" in text:
        print("    RAG: arte_test.pdf trovato nelle citazioni!")
    print(f"    SSE stream OK, {len(text)} chars")


test("POST /api/chat SSE (con RAG)", t_chat_rag)


def t_chat_generic():
    r = requests.post(
        f"{BASE}/api/chat",
        json={
            "query": "Ciao, come stai?",
            "modalita": "generale",
            "history": [],
            "stream": True,
        },
        timeout=30,
    )
    assert r.status_code == 200
    assert "event: done" in r.text


test("POST /api/chat (query generica)", t_chat_generic)


# ── Delete ──────────────────────────────────────────────────
print("\n--- Delete ---")


def t_delete():
    r = requests.delete(f"{BASE}/api/documents/arte_test.pdf")
    assert r.status_code == 200, f"status {r.status_code}"
    d = r.json()
    print(f"    {d['message']}")


test("DELETE /api/documents/arte_test.pdf", t_delete)


def t_verify_delete():
    r = requests.get(f"{BASE}/api/documents")
    d = r.json()
    fnames = [doc["filename"] for doc in d["documents"]]
    assert "arte_test.pdf" not in fnames, f"File still present: {fnames}"
    print(f"    docs rimasti: {d['total']}")


test("GET /api/documents (verifica eliminazione)", t_verify_delete)


# ── Static Assets ──────────────────────────────────────────
print("\n--- Static Assets ---")


def t_css():
    r = requests.get(f"{BASE}/css/style.css")
    assert r.status_code == 200, f"status {r.status_code}"
    assert len(r.text) > 100, f"CSS too short: {len(r.text)} chars"
    print(f"    CSS: {len(r.text)} chars")


test("GET /css/style.css", t_css)


def t_js():
    r = requests.get(f"{BASE}/js/app.js")
    assert r.status_code == 200, f"status {r.status_code}"
    assert len(r.text) > 100, f"JS too short: {len(r.text)} chars"
    print(f"    JS: {len(r.text)} chars")


test("GET /js/app.js", t_js)


# ── Riepilogo ──────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RIEPILOGO: {ok} PASSED, {fail} FAILED su {ok + fail} test")
print("=" * 60)
sys.exit(1 if fail > 0 else 0)
