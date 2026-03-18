"""
Context builder: assembla il contesto per il LLM rispettando un budget di token.
Gestisce sliding window per conversazioni multi-turn e deduplicazione.
"""

import logging
from typing import Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Assembla contesto ottimale per il LLM:
    1. Deduplica chunk simili
    2. Rispetta il budget di token
    3. Aggiunge history (sliding window) per conversazioni multi-turn
    4. Formatta con citazioni inline
    5. Supporta modalità comprehensive per analisi completa documenti
    """

    def __init__(self, max_context_tokens: Optional[int] = None):
        self.max_context_tokens = max_context_tokens or (settings.LLM_MAX_TOKENS * 2)
        # Budget espanso per analisi comprensiva (8x il normale per copertura massima)
        self.comprehensive_context_tokens = max(self.max_context_tokens * 8, 32768)

    def build(
        self,
        search_results: list,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "",
        doc_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """
        Costruisce il prompt completo per il LLM.

        Args:
            search_results: risultati re-rankati
            query: domanda utente
            history: lista di {role, content} per sliding window
            system_prompt: prompt di sistema
            doc_metadata: metadati dei documenti caricati [{source_file, total_pages, chunk_count}]
        Returns:
            Prompt completo assemblato
        """
        # Budget token: system_prompt + history + context + query
        token_budget = self.max_context_tokens
        used_tokens = self._count_tokens(system_prompt) + self._count_tokens(query)

        # --- History (sliding window: ultime N coppie) ---
        history_text = ""
        if history:
            history_text = self._format_history(history, max_tokens=token_budget // 4)
            used_tokens += self._count_tokens(history_text)

        # --- Metadati documenti ---
        metadata_text = ""
        if doc_metadata:
            metadata_text = self._format_doc_metadata(doc_metadata)
            used_tokens += self._count_tokens(metadata_text)

        # --- Contesto documenti (deduplicato) ---
        context_budget = token_budget - used_tokens - 300  # margine
        context_text = self._assemble_context(search_results, context_budget)

        # --- Assemblaggio prompt finale ---
        parts = []
        if metadata_text:
            parts.append("METADATI DOCUMENTI CARICATI:\n" + metadata_text)
        if context_text:
            parts.append("CONTESTO DAI DOCUMENTI (estratti testuali):\n" + context_text)
        if history_text:
            parts.append("CONVERSAZIONE PRECEDENTE:\n" + history_text)
        parts.append(f"DOMANDA DELL'UTENTE:\n{query}")
        parts.append(
            "ISTRUZIONI VINCOLANTI:\n"
            "1. Rispondi basandoti sul contesto fornito sopra. "
            "Se il contesto contiene informazioni ANCHE PARZIALI o INDIRETTE "
            "sull'argomento, USALE per costruire una risposta utile.\n"
            "2. MENZIONA SEMPRE esplicitamente i termini chiave dalla domanda "
            "dell'utente nella tua risposta (nomi propri, concetti, periodi).\n"
            "3. Rispondi \"Il contesto dei documenti non contiene questa informazione.\" "
            "SOLO se non c'è NESSUN riferimento utile all'argomento nel contesto.\n"
            "4. Cita SEMPRE le fonti con: [📄 nomefile.pdf | p. XX]\n"
            "5. NON inventare informazioni non presenti nel contesto.\n"
            "6. Per domande SUL file (pagine, tipo, nome), usa i METADATI DOCUMENTI CARICATI.\n"
            "7. Per domande sul CONTENUTO del file, usa il CONTESTO DAI DOCUMENTI.\n"
            "8. Sii conciso, preciso e rispondi SOLO a ciò che è stato chiesto."
        )

        return "\n\n".join(parts)

    def build_simple(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Prompt senza contesto documenti (per query generali)."""
        parts = []
        if history:
            history_text = self._format_history(history, max_tokens=self.max_context_tokens // 3)
            if history_text:
                parts.append("CONVERSAZIONE PRECEDENTE:\n" + history_text)
        parts.append(f"DOMANDA:\n{query}")
        return "\n\n".join(parts)

    def build_comprehensive(
        self,
        all_chunks: list,
        query: str,
        source_file: str,
        total_pages: int,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = "",
        doc_metadata: Optional[List[Dict]] = None,
    ) -> str:
        """
        Costruisce un prompt per analisi COMPRENSIVA di un intero documento.
        Seleziona chunk distribuiti su TUTTE le pagine per garantire copertura totale.

        Args:
            all_chunks: TUTTI i chunk del documento (ordinati per pagina)
            query: domanda utente
            source_file: nome del file sorgente
            total_pages: numero totale di pagine
            history: conversazione precedente
            system_prompt: prompt di sistema
            doc_metadata: metadati documenti
        Returns:
            Prompt completo con copertura distribuita su tutte le pagine
        """
        token_budget = self.comprehensive_context_tokens
        used_tokens = self._count_tokens(system_prompt) + self._count_tokens(query)

        # History (porzione ridotta per dare spazio al contesto)
        history_text = ""
        if history:
            history_text = self._format_history(history, max_tokens=token_budget // 8)
            used_tokens += self._count_tokens(history_text)

        # Metadati documenti
        metadata_text = ""
        if doc_metadata:
            metadata_text = self._format_doc_metadata(doc_metadata)
            used_tokens += self._count_tokens(metadata_text)

        # Seleziona chunk con copertura distribuita su TUTTE le pagine
        context_budget = token_budget - used_tokens - 500  # margine
        selected_chunks = self._select_distributed_chunks(all_chunks, context_budget)

        # Formatta contesto con citazioni per pagina
        context_text = self._format_comprehensive_context(selected_chunks, source_file)

        # Stats per logging
        pages_covered = set()
        for c in selected_chunks:
            pages_covered.add(c.get("page_number", 0))
        logger.info(
            f"Contesto comprensivo: {len(selected_chunks)}/{len(all_chunks)} chunk, "
            f"pagine coperte: {len(pages_covered)}/{total_pages}"
        )

        # Assemblaggio prompt
        parts = []
        if metadata_text:
            parts.append("METADATI DOCUMENTI CARICATI:\n" + metadata_text)

        parts.append(
            f"ANALISI COMPRENSIVA DEL DOCUMENTO: {source_file}\n"
            f"Pagine totali: {total_pages} | Chunk totali: {len(all_chunks)} | "
            f"Chunk selezionati: {len(selected_chunks)} | "
            f"Pagine coperte: {len(pages_covered)}/{total_pages}"
        )

        if context_text:
            parts.append(
                "CONTESTO COMPLETO DAI DOCUMENTI (estratti distribuiti su tutte le pagine):\n"
                + context_text
            )
        if history_text:
            parts.append("CONVERSAZIONE PRECEDENTE:\n" + history_text)
        parts.append(f"DOMANDA DELL'UTENTE:\n{query}")
        parts.append(
            "ISTRUZIONI PER ANALISI COMPRENSIVA:\n"
            "1. Analizza TUTTO il contenuto fornito, sezione per sezione.\n"
            "2. Copri TUTTI i temi principali presenti nel documento, dall'inizio alla fine.\n"
            "3. Per ogni macro-argomento, fornisci una sintesi strutturata con i punti chiave.\n"
            "4. Cita SEMPRE le fonti con: [📄 nomefile.pdf | p. XX]\n"
            "5. Organizza la risposta in modo gerarchico: titoli → sottotitoli → dettagli.\n"
            "6. NON limitarti alle prime pagine: copri anche la seconda metà del documento.\n"
            "7. Se il documento contiene domande d'esame o sezioni di verifica, elencale.\n"
            "8. Fornisci una risposta ESAUSTIVA e COMPLETA."
        )

        return "\n\n".join(parts)

    def _select_distributed_chunks(
        self, all_chunks: list, token_budget: int
    ) -> list:
        """
        Seleziona chunk distribuiti uniformemente su tutte le pagine.
        PRIORITÀ: copertura massima (ogni pagina almeno 1 chunk).
        STRATEGIA: nella prima passata, per ogni pagina scegli il chunk PIÙ CORTO
        che contenga almeno 100 caratteri (per massimizzare le pagine coperte).
        Nelle passate successive, aggiungi chunk addizionali dal più informativo.
        """
        if not all_chunks or token_budget <= 0:
            return []

        # Raggruppa chunk per pagina
        pages: Dict[int, list] = {}
        for chunk in all_chunks:
            page = chunk.get("page_number", 0)
            if page not in pages:
                pages[page] = []
            pages[page].append(chunk)

        sorted_page_nums = sorted(pages.keys())
        total_pages = len(sorted_page_nums)

        if total_pages == 0:
            return []

        # Pre-ordina i chunk di ogni pagina: dal più corto al più lungo (per pass 1)
        # ma con almeno 100 caratteri per essere informativi
        pages_by_shortest = {}
        for page_num in sorted_page_nums:
            viable = [c for c in pages[page_num] if len(c.get("text", "")) >= 100]
            if not viable:
                viable = pages[page_num]  # fallback: prendi qualsiasi chunk
            pages_by_shortest[page_num] = sorted(viable, key=lambda c: len(c.get("text", "")))

        selected = []
        selected_ids = set()
        tokens_used = 0

        # Pass 1: 1 chunk per pagina (il più corto viabile) — COPERTURA MASSIMA
        for page_num in sorted_page_nums:
            candidates = pages_by_shortest.get(page_num, [])
            if not candidates:
                continue
            candidate = candidates[0]
            cid = candidate.get("chunk_id", id(candidate))
            if cid in selected_ids:
                continue
            text = candidate.get("text", "")
            chunk_tokens = self._count_tokens(text) + 30  # overhead citazione
            if tokens_used + chunk_tokens <= token_budget:
                selected.append(candidate)
                selected_ids.add(cid)
                tokens_used += chunk_tokens

        # Pass 2+: aggiungi chunk aggiuntivi distribuiti (round-robin, dal più lungo)
        # Questa volta ordiniamo per lunghezza DECRESCENTE per aggiungere profondità
        pass_index = 0
        page_extra_idx = {p: 0 for p in sorted_page_nums}
        while tokens_used < token_budget * 0.95:
            added_any = False
            for page_num in sorted_page_nums:
                all_page = pages[page_num]
                # Cerca il prossimo chunk non ancora selezionato
                while page_extra_idx[page_num] < len(all_page):
                    candidate = all_page[page_extra_idx[page_num]]
                    cid = candidate.get("chunk_id", id(candidate))
                    page_extra_idx[page_num] += 1
                    if cid not in selected_ids:
                        text = candidate.get("text", "")
                        chunk_tokens = self._count_tokens(text) + 30
                        if tokens_used + chunk_tokens <= token_budget:
                            selected.append(candidate)
                            selected_ids.add(cid)
                            tokens_used += chunk_tokens
                            added_any = True
                        break

            if not added_any:
                break

        # Riordina per pagina e chunk_index
        selected.sort(key=lambda c: (c.get("page_number", 0), c.get("chunk_index", 0)))
        return selected

    def _format_comprehensive_context(self, chunks: list, source_file: str) -> str:
        """Formatta i chunk selezionati raggruppandoli per pagina."""
        if not chunks:
            return ""

        parts = []
        current_page = None

        for chunk in chunks:
            page = chunk.get("page_number", "?")
            text = chunk.get("text", "")

            if page != current_page:
                if current_page is not None:
                    parts.append("")  # separatore tra pagine
                parts.append(f"── Pagina {page} ──")
                current_page = page

            citation = f"[📄 {source_file} | p. {page}]"
            parts.append(f"{citation}\n{text}")

        return "\n\n".join(parts)

    def _assemble_context(self, results: list, token_budget: int) -> str:
        """Assembla contesto dai risultati, deduplicando e rispettando il budget."""
        if not results or token_budget <= 0:
            return ""

        seen_texts = set()
        context_parts = []
        tokens_used = 0

        for r in results:
            # Deduplicazione: salta chunk quasi identici
            text_key = r.text[:200].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Formatta con citazione
            citation = f"[📄 {r.source_file} | p. {r.page_number}]"
            formatted = f"{citation}\n{r.text}"

            chunk_tokens = self._count_tokens(formatted)
            if tokens_used + chunk_tokens > token_budget:
                # Se non c'è spazio per il chunk completo, tronca
                remaining = token_budget - tokens_used
                if remaining > 100:
                    truncated = formatted[: remaining * 4]  # approssimazione
                    context_parts.append(truncated + "…")
                break

            context_parts.append(formatted)
            tokens_used += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def _format_history(self, history: List[Dict[str, str]], max_tokens: int) -> str:
        """
        Formatta la history con sliding window.
        Mantiene le interazioni più recenti entro il budget.
        """
        if not history:
            return ""

        # Parti dalle più recenti e aggiungi finché c'è spazio
        formatted_parts = []
        tokens = 0

        for msg in reversed(history):
            role = "Tu" if msg.get("role") == "user" else "Assistente"
            line = f"{role}: {msg.get('content', '')}"
            line_tokens = self._count_tokens(line)

            if tokens + line_tokens > max_tokens:
                break
            formatted_parts.insert(0, line)
            tokens += line_tokens

        return "\n".join(formatted_parts)

    @staticmethod
    def _format_doc_metadata(doc_metadata: List[Dict]) -> str:
        """Formatta i metadati dei documenti caricati."""
        lines = []
        for doc in doc_metadata:
            name = doc.get("source_file", "sconosciuto")
            pages = doc.get("total_pages", "N/D")
            chunks = doc.get("chunk_count", "N/D")
            lines.append(
                f"- File: {name} | Pagine totali reali: {pages} | Chunk indicizzati: {chunks}"
            )
        return "\n".join(lines)

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Stima token: ~4 caratteri per token (approssimazione per italiano)."""
        return max(1, len(text) // 4)
