# Decisioni Architetturali — IAGestioneArte

Documento che spiega le scelte tecniche principali e le motivazioni.

---

## 1. Embedding: multilingual-e5-base vs all-MiniLM-L6-v2

**Scelta**: `intfloat/multilingual-e5-base` (768 dimensioni)

**Motivazione**:
- MiniLM-L6-v2 è ottimizzato per inglese → scarsa performance su testi italiani di storia dell'arte
- E5-base è addestrato su 100+ lingue con fine-tuning su retrieval tasks
- Il prefisso `query: ` / `passage: ` permette al modello di distinguere il ruolo del testo
- 768 dimensioni offrono un buon equilibrio qualità/velocità

**Alternativa**: `intfloat/multilingual-e5-large` (1024d) per massima qualità, configurabile in `config.yaml`

---

## 2. Ricerca Ibrida: Dense + BM25 + RRF

**Scelta**: Combinare ricerca vettoriale (ChromaDB) con BM25 (rank_bm25) tramite Reciprocal Rank Fusion

**Motivazione**:
- La ricerca solo vettoriale fallisce su keyword specifiche (nomi propri, date, codici)
- BM25 cattura match esatti che gli embeddings possono perdere
- RRF (k=60) combina i ranking senza richiedere calibrazione dei punteggi
- Formula: `RRF(d) = Σ 1/(k + rank_i(d))`

**Implementazione**:
- BM25 ricostruito in memoria da ChromaDB all'avvio (veloce, <1s per 10K chunks)
- ChromaDB è la fonte di verità persistente
- Tokenizzazione BM25: lowercase + split su non-alfanumerici

---

## 3. Re-ranking con Cross-Encoder

**Scelta**: `cross-encoder/ms-marco-MiniLM-L-6-v2` locale + Cohere API come alternativa

**Motivazione**:
- Il re-ranking cross-encoder migliora la precisione del 15-25% nei benchmark
- Il modello locale è leggero (~80MB) e non richiede API key
- Cohere `rerank-multilingual-v3.0` è l'alternativa cloud più performante per italiano

**Trade-off**: ~200ms di latenza aggiuntiva per il re-ranking locale, accettabile per UX

---

## 4. Chunking Semantico vs Token-Based

**Scelta**: Chunking basato su frasi (NLTK Italian sentence tokenizer) con overlap per frasi

**Motivazione**:
- Il chunking a token fissi taglia frasi a metà → perdita di senso
- NLTK `punkt_tab` per italiano rispetta i confini delle frasi
- L'overlap è calcolato in frasi (non token) per mantenere coerenza
- Le tabelle sono trattate come chunk atomici (non frammentate)

**Parametri default**: 512 token/chunk, 50 token overlap → configurabili in `config.yaml`

---

## 5. LLM: 8-Tier Fallback Chain

**Scelta**: Cascata di provider LLM con fallback automatico

**Motivazione**:
- Nessun singolo provider garantisce uptime 100%
- L'ordine è ottimizzato per: qualità → velocità → costo
- Ogni backend è a import lazy: se il SDK non è installato, si salta silenziosamente
- L'utente non percepisce i fallback (trasparente)

**Ordine**: OpenRouter > Groq > Mistral > Gemini > HuggingFace > DeepSeek > Ollama

---

## 6. FastAPI vs Streamlit

**Scelta**: FastAPI + SPA HTML/CSS/JS

**Motivazione**:
- Streamlit è limitato per UX personalizzata (WCAG, SSE, tooltips)
- FastAPI offre controllo completo su API REST + SSE + static files
- Il frontend SPA è leggero (nessun framework JS) e compliant WCAG AA
- La stessa app serve API + UI → un singolo processo da deployare

---

## 7. SSE vs WebSocket per Streaming

**Scelta**: Server-Sent Events (SSE)

**Motivazione**:
- SSE è unidirezionale (server → client): perfetto per streaming risposte
- Più semplice di WebSocket: nessun handshake, auto-reconnect nativo
- Funziona attraverso proxy e CDN senza configurazione speciale
- Sufficiente per il nostro caso d'uso (non serve bidirezionalità)

---

## 8. Token Budget nel Context Builder

**Scelta**: Assemblaggio contesto con budget massimo di token

**Motivazione**:
- Ogni modello LLM ha un context window limitato
- Senza budget, troppi chunk portano a rigetto o troncamento silenzioso
- Il budget funziona così:
  1. Primi N token per system prompt
  2. Poi sliding window per la cronologia (più recente prima)
  3. Poi documenti rilevanti in ordine di score fino a esaurimento budget
- Stima ~4 char/token (pragmatica, evita dipendenza da tiktoken per ogni provider)

---

## 9. Error Handling: Nessun Stack Trace all'Utente

**Scelta**: Middleware globale che converte tutte le eccezioni in messaggi umani in italiano

**Motivazione**:
- Gli stack trace sono inutili e spaventosi per l'utente finale
- Ogni tipo di errore ha un messaggio + suggerimento in italiano
- I dettagli tecnici vanno solo nei log (rotazione automatica)

---

## 10. Struttura Modulare senza Circular Imports

**Scelta**: Dipendenze unidirezionali: `config → ingestion → retrieval → llm → api`

**Motivazione**:
- Nessun modulo importa dal livello superiore
- I tipi (dataclass) sono definiti dove vengono creati
- Le init `__init__.py` ri-espongono solo le classi pubbliche
- `api/routes.py` usa `app.state` per condividere i servizi (dependency injection primitiva)
