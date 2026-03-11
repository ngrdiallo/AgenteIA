# IAGestioneArte

**Assistente AI RAG avanzato per la gestione dei Beni Culturali — ABA Bari**

Sistema di Retrieval-Augmented Generation con ricerca ibrida (dense + BM25), re-ranking cross-encoder, citazioni precise e interfaccia web accessibile WCAG AA.

---

## Quick Start

```bash
# 1. Clona e vai nella cartella
cd IAGestioneArte

# 2. Crea ambiente virtuale
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Configura le API keys
cp .env.example .env
# Modifica .env con almeno 1 API key LLM

# 5. Avvia
python main.py
```

Apri il browser su **http://localhost:8000**

---

## Architettura

```
IAGestioneArte/
├── main.py                    # Entry point
├── config.yaml                # Configurazione centralizzata
├── .env                       # API keys (non versionato)
├── requirements.txt          
│
├── config/                    # Configurazione + system prompts
│   ├── __init__.py           # Settings singleton
│   └── prompts.py            # Prompts per 8 modalità operative
│
├── ingestion/                 # Pipeline ingestione documenti
│   ├── loaders.py            # PDF, DOCX, PPTX, XLSX, Immagini + OCR
│   ├── chunker.py            # Chunking semantico con NLTK italiano
│   ├── embedder.py           # Multilingual-E5 embeddings
│   └── indexer.py            # ChromaDB persistente
│
├── retrieval/                 # Motore di ricerca ibrido
│   ├── hybrid_search.py      # Dense + BM25 + RRF fusion
│   ├── reranker.py           # Cross-encoder + Cohere
│   ├── context_builder.py    # Assemblaggio contesto con token budget
│   └── citation_manager.py   # Citazioni precise con snippet
│
├── llm/                       # Layer LLM multi-backend
│   ├── orchestrator.py       # 8-tier fallback chain
│   ├── quality_evaluator.py  # Valutazione profondità risposte
│   ├── italian_filter.py     # Filtro output italiano
│   └── vision.py             # Analisi immagini artistiche
│
├── api/                       # REST API
│   ├── models.py             # Pydantic schemas
│   ├── middleware.py         # Error handler, CORS, logging
│   └── routes.py             # Endpoints + SSE streaming
│
├── ui/                        # Frontend SPA
│   ├── index.html            # Interfaccia chat accessibile
│   ├── css/style.css         # WCAG AA, responsive
│   └── js/app.js             # Chat SSE, upload, citazioni
│
├── system/                    # Servizi di sistema
│   ├── cleanup.py            # Pulizia automatica temp/log
│   ├── logging_config.py     # Log strutturati con rotazione
│   └── health.py             # Health monitoring
│
└── tests/                     # Test suite
    └── conftest.py
```

---

## Pipeline RAG

```
Query utente
    │
    ▼
┌─────────────────────────────────┐
│  1. HYBRID SEARCH               │
│  ├─ Dense: ChromaDB + E5 embed  │
│  ├─ Sparse: BM25 tokenizzato    │
│  └─ RRF Fusion (k=60)           │
└───────────┬─────────────────────┘
            ▼
┌─────────────────────────────────┐
│  2. RERANKING                    │
│  Cross-encoder MiniLM-L-6 / Cohere│
└───────────┬─────────────────────┘
            ▼
┌─────────────────────────────────┐
│  3. CONTEXT BUILDER              │
│  Token budget + dedup + history  │
└───────────┬─────────────────────┘
            ▼
┌─────────────────────────────────┐
│  4. LLM (8-tier fallback)       │
│  OpenRouter → Groq → Mistral →  │
│  Gemini → HF → DeepSeek → Ollama│
└───────────┬─────────────────────┘
            ▼
┌─────────────────────────────────┐
│  5. POST-PROCESSING              │
│  ├─ Italian filter               │
│  ├─ Quality evaluation           │
│  └─ Citation formatting          │
└─────────────────────────────────┘
```

---

## Configurazione

### API Keys (`.env`)

Configura **almeno 1** LLM backend:

| Variabile | Provider | Note |
|-----------|----------|------|
| `OPENROUTER_API_KEY` | OpenRouter | Auto-routing, molti modelli gratuiti |
| `GROQ_API_KEY` | Groq | Ultra-veloce, Llama 3.3 70B |
| `MISTRAL_API_KEY` | Mistral | Mistral Small |
| `GOOGLE_API_KEY` | Google | Gemini 2.0 Flash Lite |
| `HF_TOKEN` | HuggingFace | Inference API |
| `DEEPSEEK_API_KEY` | DeepSeek | Chat + Vision |
| `COHERE_API_KEY` | Cohere | Reranker multilingue (opzionale) |

### Configurazione avanzata (`config.yaml`)

Tutti i parametri RAG, embedding, reranking e LLM sono configurabili senza modificare il codice.

---

## 8 Modalità Operative

| Modalità | Descrizione |
|----------|-------------|
| 🎯 Generale | Assistenza generica beni culturali |
| 🧠 Ragionamento | Analisi causale approfondita |
| 🔬 Analisi Critica | Confronti multi-prospettiva |
| 👗 Fashion Design | Storia costume e moda |
| 📝 Preparazione Esami | Quiz e schemi di studio |
| 🎤 Presentazioni | Generazione slide e discorsi |
| 📑 Analisi Documenti | Estrazione dati da documenti |
| 🎨 Storico-Artistico | Analisi opere d'arte |

---

## API Endpoints

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/api/chat` | Chat SSE streaming |
| POST | `/api/chat/sync` | Chat sincrona |
| POST | `/api/documents/upload` | Upload documento |
| GET | `/api/documents` | Lista documenti |
| DELETE | `/api/documents/{filename}` | Elimina documento |
| POST | `/api/vision/analyze` | Analisi immagine |
| GET | `/api/health` | Health check |
| GET | `/api/modalita` | Lista modalità |
| GET | `/docs` | OpenAPI docs |

---

## Formati Supportati

- **PDF** — PyMuPDF con OCR fallback per scansioni
- **DOCX** — Word con tabelle
- **PPTX** — PowerPoint slide per slide
- **XLSX** — Excel foglio per foglio
- **Immagini** — PNG, JPG, JPEG, WebP (OCR + analisi vision)

---

## Accessibilità (WCAG AA)

- Contrasto ≥ 4.5:1 su tutti gli elementi
- Font base 16px, scalabile
- Skip navigation link
- Focus visibile su tutti gli elementi interattivi  
- `aria-label` e `aria-live` per screen reader
- Supporto navigazione da tastiera completo
- Responsive fino a 320px

---

## Troubleshooting

| Problema | Soluzione |
|----------|----------|
| "Tutti i backend LLM falliti" | Verifica le API key in `.env` |
| Embedding lento al primo avvio | Download modello (~560MB), attendi |
| OCR non funziona | Installa Tesseract: `apt install tesseract-ocr tesseract-ocr-ita` |
| Porta 8000 occupata | Modifica `port` in `config.yaml` |
