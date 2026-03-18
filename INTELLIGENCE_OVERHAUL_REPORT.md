# IAGestioneArte — Report Intelligenza Profonda
## Sessione di sviluppo: 22 Febbraio 2026

---

## 🎯 OBIETTIVO
> "Ciò che stiamo creando deve essere intelligente almeno quanto te — un clone mentale."

## 📊 RISULTATI PRIMA → DOPO

| Query critica | PRIMA | DOPO |
|--------------|-------|------|
| "ciao analizza il file, quante pagine ha?" | ❌ `greeting` → "Non ho ricevuto alcun file" | ✅ `comprehensive` → 684 parole, 212 citazioni, 201/220 pagine |
| "quante pagine ha il documento?" | ❌ Nessun path dedicato → risposta vaga | ✅ PATH META → tabella 4 file con pagine reali, 2.4s |
| "chi era Bernini?" | ❌ "Il contesto non contiene questa informazione" | ✅ 336 parole, 5 citazioni da pagine reali (p.42,68,87,97,107) |
| "ciao come stai?" | ❌ `focused` → reranker → build_simple | ✅ `greeting` → tabella documenti + suggerimenti, 8.4s |
| "buonasera, dimmi tutto sul Barocco" | ❌ `greeting` → contesto vuoto | ✅ `comprehensive` → 1636 parole, copertura 202/220 pagine |
| "questo file di cosa tratta?" | ❌ `focused` → risultati scarsi | ✅ `comprehensive` → analisi completa documento principale |
| Upload duplicato stesso PDF | ❌ 4 copie senza avviso | ✅ Dedup automatica: vecchia versione eliminata |

---

## 🔧 FIX IMPLEMENTATI (9 totali)

### FIX 1: Classificatore Intelligente (RISCRITTURA COMPLETA)
**File**: `retrieval/query_classifier.py`
- **Prima**: Regex-first, greeting patterns controllati PER PRIMI → "ciao analizza il file" = greeting
- **Dopo**: Sistema multi-segnale con punteggi pesati
- **Principio**: L'intento informativo vince SEMPRE sul saluto
- **Test**: 14/14 superati

### FIX 2: PATH META per Metadati
**File**: `api/routes.py`
- Nuovo path dedicato: `query_type == "meta"` → risponde usando SOLO metadati
- "quante pagine?" → tabella con tutti i file, pagine, chunk → 2.4s

### FIX 3: Query Vaghe e Multi-intento
**File**: `retrieval/query_classifier.py`
- `_strip_greeting_prefix()`: rimuove "ciao/buongiorno" prima dell'analisi
- `_has_informative_content()`: ~35 keyword informative
- Se ANY keyword informativa presente → greeting score = 0

### FIX 4: Logging Query Completo
**File**: `api/routes.py`
- `📊 Query classificata:` ora include intenti rilevati
- `📝 Testo query:` mostra il testo effettivo della query

### FIX 5: Upload Deduplicazione
**File**: `api/routes.py`
- Controlla se file esiste già → elimina vecchia versione → re-indicizza
- Messaggio chiaro: "Documento aggiornato" vs "Documento caricato"

### FIX 6: System Prompt Proattivo
**File**: `config/prompts.py`
- 6 regole di INTELLIGENZA PROATTIVA
- Mai dire "non ho ricevuto file" se documenti esistono nel contesto
- Gestione multi-richiesta: risponde a TUTTE le domande

### FIX 7: Copertura 91% Pagine (era 36%)
**File**: `retrieval/context_builder.py`
- `comprehensive_context_tokens` = 32768 (era 16384)
- `_select_distributed_chunks()`: Pass 1 prende chunk PIÙ CORTI (≥100 char) per massimizzare copertura
- Risultato: 201/220 pagine (91%) vs 79/220 (36%)

### FIX 8: Greeting Path dentro use_rag
**File**: `api/routes.py`
- Il greeting con `use_rag=true` ora entra nel path corretto (era nel blocco `else`)
- Saluto → tabella documenti + suggerimenti personalizzati

### FIX 9: "Come stai?" non è Focused
**File**: `retrieval/query_classifier.py`
- `_GREETING_PHRASES`: "come stai", "come va", "tutto bene" esclusi dal focused
- Pattern compositi per greeting: "ciao come stai, tutto bene?"
- Test: 14/14 superati

### FIX 10: Orchestrator Intelligente
**File**: `llm/orchestrator.py`
- **Smart backend routing**: stima token del prompt, salta backend con limiti troppo bassi
- **Riordina catena per prompt grandi**: Gemini (1M) → Mistral (128K) → DeepSeek (64K) prima
- **HuggingFace**: aggiunto `provider="hf-inference"` per evitare modelli deprecati
- **ChromaDB telemetry**: silenziato errore innocuo posthog

---

## 🛡️ ANALISI ERRORI PREVENTIVI

### Errori Rilevati nei Test
| Backend | Errore | Causa | Stato |
|---------|--------|-------|-------|
| OpenRouter | 402 | Crediti esauriti | ⚠️ Attivo ma senza crediti |
| Groq | 413 (TPM) | Prompt 41K > limite 12K | ✅ Ora skippato automaticamente |
| Mistral | 429 | Rate limit dopo molte richieste | ✅ Funziona, limite temporaneo |
| Gemini | 429 | Quota giornaliera free tier esaurita | ⚠️ Si resetta domani |
| HuggingFace | 410 | Modello deprecato su provider "together" | ✅ Corretto con provider="hf-inference" |
| DeepSeek | 402 | Saldo insufficiente | ⚠️ Richiede ricarica |

### Errori Potenziali da Prevenire
1. **PDF corrotto**: Se un PDF non è parsabile, `loader.load()` potrebbe crashare → suggerimento: aggiungere try/catch con messaggio utente
2. **Upload concorrente**: Due upload simultanei dello stesso file → possibile race condition nella dedup
3. **ChromaDB corruption**: Se il database è corrotto → aggiungere health check periodico
4. **Query molto lunga (>10K caratteri)**: Potrebbe causare timeout → suggerimento: troncamento intelligente
5. **Session store overflow**: 100+ sessioni con migliaia di messaggi → suggerimento: pulizia automatica
6. **Embedding model failure**: Se il modello non si scarica → timeout + messaggio chiaro

---

## 📋 PIANO DI SVILUPPO FUTURO

### PRIORITÀ 1 — Immediato (prossima sessione)
- [ ] **Ricaricare crediti API**: OpenRouter, DeepSeek, Gemini
- [ ] **Aggiungere caching LLM**: risposte identiche alla stessa query = cache hit (Redis o semplice dict)
- [ ] **Warmup modelli all'avvio**: caricare embedding + cross-encoder durante startup (non alla prima query)

### PRIORITÀ 2 — A breve (1-2 sessioni)
- [ ] **Streaming SSE completo**: la route `/api/chat` SSE deve usare lo stesso flusso di `/api/chat/sync`
- [ ] **Retry con backoff**: per errori 429/rate-limit, attendere e riprovare (Gemini suggerisce retry_delay)
- [ ] **PDF corruption handling**: try/catch in ingestion con messaggio specifico
- [ ] **Query history context**: usare le ultime N query della sessione come contesto aggiuntivo
- [ ] **Concurrent upload protection**: lock per file per evitare race condition

### PRIORITÀ 3 — Miglioramenti (3-5 sessioni)
- [ ] **Multi-file comprehensive**: supportare analisi che copre TUTTI i documenti, non solo il più grande
- [ ] **Chunk-level caching**: memorizzare i chunk embedding per velocizzare re-indexing
- [ ] **Adaptive token budget**: basare il budget token sulla lunghezza reale dei chunk, non stima fissa
- [ ] **UI migliorata**: mostrare le citazioni come link cliccabili con preview del testo
- [ ] **Export risposte**: download risposta come PDF/DOCX con citazioni formattate

### PRIORITÀ 4 — Evoluzione sistema
- [ ] **Multi-lingua detection**: identificare la lingua della query e rispondere nella stessa
- [ ] **Agent memory**: ricordare preferenze utente tra sessioni
- [ ] **Auto-evaluation**: valutare la qualità della risposta e riformulare se troppo bassa
- [ ] **A/B testing backend**: testare quale backend produce risposte migliori per tipo di query

---

## 🏗️ ARCHITETTURA ATTUALE

```
QUERY → Classificatore Multi-Segnale → 4 PATH
  │
  ├─ META     → Metadati documenti (no RAG, 2-3s)
  ├─ GREETING → Saluto + lista documenti (no RAG, 8s)
  ├─ COMPREHENSIVE → Tutti i chunk distribuiti → 91% copertura → LLM 4096 token (15-20s)
  └─ FOCUSED  → Hybrid search → Dual rerank → Top-5 chunks → LLM 2048 token (5-7s)

LLM ORCHESTRATOR:
  Stima token → [Se grande: Gemini→Mistral→DeepSeek] | [Se piccolo: OpenRouter→Groq→...]
  Skip backend con limiti insufficienti → Fallback a cascata → Risposta o errore gentile
```

---

## ✅ TEST SUPERATI
- 14/14 test unitari classificatore
- 5/5 tipologie query live (comprehensive, focused, meta, greeting, greeting+docs)
- Fallback multi-backend verificato con 6 diversi errori API
- Upload dedup verificato
- Smart backend routing per prompt grandi

---

*Server attivo su porta 8001 | 4 documenti indicizzati | 6 backend configurati*
