/**
 * IAGestioneArte — Client JavaScript
 * Chat SSE, upload con progress, tooltip citazioni, sessioni persistenti.
 */

(function () {
    "use strict";

    const API = "";  // Same-origin, no prefix

    // ── DOM refs ────────────────────────────────────────

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const onboarding = $("#onboarding");
    const chatSection = $("#chat-section");
    const messages = $("#messages");
    const chatInput = $("#chat-input");
    const btnSend = $("#btn-send");
    const btnStart = $("#btn-start");
    const btnDocs = $("#btn-docs");
    const btnCloseDocs = $("#btn-close-docs");
    const docsPanel = $("#docs-panel");
    const docList = $("#doc-list");
    const dropZone = $("#drop-zone");
    const fileInput = $("#file-input");
    const chatFileInput = $("#chat-file-input");
    const btnUpload = $("#btn-upload");
    const uploadProgress = $("#upload-progress");
    const progressFill = $("#progress-fill");
    const uploadStatus = $("#upload-status");
    const charCount = $("#char-count");
    const modalitaSelect = $("#modalita-select");
    const toastContainer = $("#toast-container");

    // Sessions panel
    const btnSessions = $("#btn-sessions");
    const sessionsPanel = $("#sessions-panel");
    const btnCloseSessions = $("#btn-close-sessions");
    const btnNewChat = $("#btn-new-chat");
    const sessionList = $("#session-list");

    let conversationHistory = [];
    let isWaiting = false;
    let currentSessionId = null;

    // ── Init ────────────────────────────────────────────

    function init() {
        if (localStorage.getItem("ia_onboarded")) {
            showChat();
        }

        bindEvents();
        loadDocuments();
        loadSessions();

        // Ripristina ultima sessione attiva
        const lastSession = localStorage.getItem("ia_current_session");
        if (lastSession) {
            loadSession(lastSession);
        }
    }

    function showChat() {
        onboarding.classList.add("hidden");
        chatSection.classList.remove("hidden");
    }

    function bindEvents() {
        btnStart.addEventListener("click", () => {
            localStorage.setItem("ia_onboarded", "1");
            showChat();
        });

        // Chat
        btnSend.addEventListener("click", sendMessage);
        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        chatInput.addEventListener("input", handleInputChange);

        // Documents panel
        btnDocs.addEventListener("click", () => {
            docsPanel.classList.toggle("hidden");
            sessionsPanel.classList.add("hidden");
        });
        btnCloseDocs.addEventListener("click", () => docsPanel.classList.add("hidden"));

        // Sessions panel
        btnSessions.addEventListener("click", () => {
            sessionsPanel.classList.toggle("hidden");
            docsPanel.classList.add("hidden");
            loadSessions();
        });
        btnCloseSessions.addEventListener("click", () => sessionsPanel.classList.add("hidden"));
        btnNewChat.addEventListener("click", startNewChat);

        // Upload from chat
        btnUpload.addEventListener("click", () => chatFileInput.click());
        chatFileInput.addEventListener("change", (e) => handleFiles(e.target.files));

        // Drop zone
        dropZone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => handleFiles(e.target.files));
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            handleFiles(e.dataTransfer.files);
        });

        // Keyboard: drop zone enter/space
        dropZone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                fileInput.click();
            }
        });
    }

    // ── Sessions ────────────────────────────────────────

    async function loadSessions() {
        try {
            const res = await fetch(`${API}/api/sessions`);
            if (!res.ok) return;
            const data = await res.json();

            if (!data.sessions || data.sessions.length === 0) {
                sessionList.innerHTML = '<p class="empty-state">Nessuna conversazione salvata.</p>';
                return;
            }

            sessionList.innerHTML = "";
            data.sessions.forEach((s) => {
                const item = document.createElement("div");
                item.className = "session-item" + (s.session_id === currentSessionId ? " active" : "");
                item.setAttribute("role", "listitem");

                const dateStr = s.updated_at ? new Date(s.updated_at).toLocaleString("it-IT", {
                    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit"
                }) : "";

                item.innerHTML = `
                    <div class="session-info" data-id="${escapeHtml(s.session_id)}">
                        <span class="session-title">${escapeHtml(s.title)}</span>
                        <span class="session-meta">${s.message_count} msg — ${dateStr}</span>
                    </div>
                    <button class="btn btn-icon session-delete" data-id="${escapeHtml(s.session_id)}" aria-label="Elimina conversazione" title="Elimina">🗑️</button>
                `;

                item.querySelector(".session-info").addEventListener("click", () => {
                    loadSession(s.session_id);
                    sessionsPanel.classList.add("hidden");
                });

                item.querySelector(".session-delete").addEventListener("click", async (e) => {
                    e.stopPropagation();
                    if (confirm("Eliminare questa conversazione?")) {
                        await deleteSession(s.session_id);
                    }
                });

                sessionList.appendChild(item);
            });
        } catch (err) {
            // Silently fail
        }
    }

    async function loadSession(sessionId) {
        try {
            const res = await fetch(`${API}/api/sessions/${sessionId}`);
            if (!res.ok) {
                if (res.status === 404) {
                    localStorage.removeItem("ia_current_session");
                    currentSessionId = null;
                }
                return;
            }
            const session = await res.json();

            currentSessionId = session.session_id;
            localStorage.setItem("ia_current_session", currentSessionId);

            // Set modalità
            if (session.modalita && modalitaSelect) {
                modalitaSelect.value = session.modalita;
            }

            // Rebuild history and UI
            conversationHistory = [];
            messages.innerHTML = "";

            if (session.messages.length === 0) {
                messages.innerHTML = `<div class="welcome-message">
                    <p>👋 Ciao! Sono il tuo assistente per i Beni Culturali.</p>
                    <p>Carica dei documenti e fammi delle domande, oppure chiedimi direttamente.</p>
                </div>`;
            } else {
                session.messages.forEach((msg) => {
                    if (msg.role === "user") {
                        appendMessage("user", msg.content);
                        conversationHistory.push({ role: "user", content: msg.content });
                    } else if (msg.role === "assistant") {
                        const citations = (msg.metadata && msg.metadata.citations) || [];
                        const backend = (msg.metadata && msg.metadata.backend) || "";
                        const latency = (msg.metadata && msg.metadata.latency_s) || 0;
                        appendAssistantMessage({
                            text: msg.content,
                            citations: citations,
                            backend: backend,
                            latency_s: latency,
                        });
                        conversationHistory.push({ role: "assistant", content: msg.content });
                    }
                });
            }

            showChat();
        } catch (err) {
            showToast("Errore nel caricamento della conversazione.", "error");
        }
    }

    async function startNewChat() {
        try {
            const res = await fetch(`${API}/api/sessions`, { method: "POST" });
            if (!res.ok) throw new Error("Errore creazione sessione");
            const data = await res.json();

            currentSessionId = data.session_id;
            localStorage.setItem("ia_current_session", currentSessionId);
            conversationHistory = [];

            messages.innerHTML = `<div class="welcome-message">
                <p>👋 Ciao! Sono il tuo assistente per i Beni Culturali.</p>
                <p>Carica dei documenti e fammi delle domande, oppure chiedimi direttamente.</p>
            </div>`;

            sessionsPanel.classList.add("hidden");
            showChat();
            showToast("Nuova conversazione creata.", "success");
        } catch (err) {
            showToast("Errore creazione nuova chat.", "error");
        }
    }

    async function deleteSession(sessionId) {
        try {
            const res = await fetch(`${API}/api/sessions/${sessionId}`, { method: "DELETE" });
            if (res.ok) {
                showToast("Conversazione eliminata.", "success");
                if (sessionId === currentSessionId) {
                    currentSessionId = null;
                    localStorage.removeItem("ia_current_session");
                    conversationHistory = [];
                    messages.innerHTML = `<div class="welcome-message">
                        <p>👋 Ciao! Sono il tuo assistente per i Beni Culturali.</p>
                        <p>Carica dei documenti e fammi delle domande, oppure chiedimi direttamente.</p>
                    </div>`;
                }
                loadSessions();
            }
        } catch (err) {
            showToast("Errore durante l'eliminazione.", "error");
        }
    }

    // ── Input handling ──────────────────────────────────

    function handleInputChange() {
        const len = chatInput.value.length;
        charCount.textContent = `${len}/5000`;
        btnSend.disabled = len === 0 || isWaiting;
        // Auto-resize
        chatInput.style.height = "auto";
        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
    }

    // ── Send message ────────────────────────────────────

    async function sendMessage() {
        const query = chatInput.value.trim();
        if (!query || isWaiting) return;

        // Auto-create session if none exists
        if (!currentSessionId) {
            try {
                const res = await fetch(`${API}/api/sessions`, { method: "POST" });
                if (res.ok) {
                    const data = await res.json();
                    currentSessionId = data.session_id;
                    localStorage.setItem("ia_current_session", currentSessionId);
                }
            } catch (err) {
                // Continue without session
            }
        }

        isWaiting = true;
        btnSend.disabled = true;
        chatInput.value = "";
        handleInputChange();

        // Add user message
        appendMessage("user", query);
        conversationHistory.push({ role: "user", content: query });

        // Show thinking
        const thinkingEl = appendThinking();

        try {
            const response = await fetchSSE(query);
            thinkingEl.remove();
            appendAssistantMessage(response);
            conversationHistory.push({ role: "assistant", content: response.text });

            // Update session_id if returned
            if (response.session_id) {
                currentSessionId = response.session_id;
                localStorage.setItem("ia_current_session", currentSessionId);
            }
        } catch (err) {
            thinkingEl.remove();
            appendMessage("assistant", "Mi dispiace, si è verificato un errore. Riprova tra qualche istante.");
            showToast(err.message || "Errore di connessione", "error");
        }

        isWaiting = false;
        btnSend.disabled = chatInput.value.length === 0;
        chatInput.focus();
    }

    // ── SSE fetch ───────────────────────────────────────

    function fetchSSE(query) {
        return new Promise((resolve, reject) => {
            const payload = {
                query: query,
                modalita: modalitaSelect.value,
                use_rag: true,
                session_id: currentSessionId,
                history: conversationHistory.slice(-10),
            };

            fetch(`${API}/api/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            })
                .then((res) => {
                    if (!res.ok) throw new Error("Errore del server");
                    return res.body.getReader();
                })
                .then((reader) => {
                    const decoder = new TextDecoder();
                    let buffer = "";
                    let result = { text: "", citations: [], backend: "", model: "", latency_s: 0, session_id: null };

                    function read() {
                        reader.read().then(({ done, value }) => {
                            if (done) {
                                resolve(result);
                                return;
                            }

                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split("\n");
                            buffer = lines.pop() || "";

                            for (const line of lines) {
                                if (line.startsWith("event: ")) {
                                    result._currentEvent = line.slice(7).trim();
                                } else if (line.startsWith("data: ")) {
                                    try {
                                        const data = JSON.parse(line.slice(6));
                                        processSSEEvent(result._currentEvent, data, result);
                                    } catch (e) {
                                        // ignore parse errors
                                    }
                                }
                            }

                            read();
                        }).catch(reject);
                    }

                    read();
                })
                .catch(reject);
        });
    }

    function processSSEEvent(event, data, result) {
        switch (event) {
            case "sources":
                result.citations = data.citations || [];
                break;
            case "answer":
                result.text = data.text || "";
                result.backend = data.backend || "";
                result.model = data.model || "";
                result.latency_s = data.latency_s || 0;
                if (data.session_id) result.session_id = data.session_id;
                break;
            case "quality":
                result.quality = data;
                break;
            case "error":
                result.text = data.error || "Errore durante l'elaborazione.";
                break;
        }
    }

    // ── Render messages ─────────────────────────────────

    function appendMessage(role, text) {
        const welcome = messages.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        const msg = document.createElement("div");
        msg.className = `message message-${role}`;

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.setAttribute("aria-hidden", "true");
        avatar.textContent = role === "user" ? "👤" : "🏛️";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.textContent = text;

        msg.appendChild(avatar);
        msg.appendChild(bubble);
        messages.appendChild(msg);
        scrollToBottom();
    }

    function appendAssistantMessage(response) {
        const welcome = messages.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        const msg = document.createElement("div");
        msg.className = "message message-assistant";

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.setAttribute("aria-hidden", "true");
        avatar.textContent = "🏛️";

        const content = document.createElement("div");

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.innerHTML = formatText(response.text);

        if (response.citations && response.citations.length > 0) {
            const citationsEl = document.createElement("div");
            citationsEl.className = "message-citations";
            citationsEl.style.marginTop = "8px";

            response.citations.forEach((c) => {
                const link = document.createElement("a");
                link.className = "citation-link";
                link.href = "#";
                link.textContent = `📄 ${c.source_file} | p. ${c.page_number}`;
                link.setAttribute("aria-label", `Citazione: ${c.source_file}, pagina ${c.page_number}`);
                link.addEventListener("click", (e) => {
                    e.preventDefault();
                    showCitationTooltip(link, c);
                });
                citationsEl.appendChild(link);
                citationsEl.appendChild(document.createTextNode(" "));
            });

            bubble.appendChild(citationsEl);
        }

        content.appendChild(bubble);

        const meta = document.createElement("div");
        meta.className = "message-meta";
        if (response.backend) {
            meta.innerHTML = `<span>🔧 ${response.backend}</span>`;
        }
        if (response.latency_s) {
            meta.innerHTML += `<span>⏱️ ${response.latency_s}s</span>`;
        }
        if (response.quality && response.quality.confidence !== undefined) {
            const conf = Math.round(response.quality.confidence * 100);
            meta.innerHTML += `<span>📊 ${conf}%</span>`;
        }
        content.appendChild(meta);

        msg.appendChild(avatar);
        msg.appendChild(content);
        messages.appendChild(msg);
        scrollToBottom();
    }

    function appendThinking() {
        const msg = document.createElement("div");
        msg.className = "message message-assistant";

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.setAttribute("aria-hidden", "true");
        avatar.textContent = "🏛️";

        const bubble = document.createElement("div");
        bubble.className = "message-bubble thinking";
        bubble.setAttribute("aria-label", "Elaborazione in corso");
        bubble.innerHTML = '<div class="thinking-dot"></div><div class="thinking-dot"></div><div class="thinking-dot"></div>';

        msg.appendChild(avatar);
        msg.appendChild(bubble);
        messages.appendChild(msg);
        scrollToBottom();
        return msg;
    }

    // ── Citation tooltip ────────────────────────────────

    function showCitationTooltip(anchor, citation) {
        document.querySelectorAll(".citation-tooltip").forEach((t) => t.remove());

        const tooltip = document.createElement("div");
        tooltip.className = "citation-tooltip visible";
        tooltip.innerHTML = `
            <div class="tooltip-header">📄 ${escapeHtml(citation.source_file)} — Pagina ${citation.page_number}</div>
            <div class="tooltip-text">${escapeHtml(citation.text_snippet || "Nessun estratto disponibile.")}</div>
        `;

        document.body.appendChild(tooltip);

        const rect = anchor.getBoundingClientRect();
        tooltip.style.top = rect.bottom + 8 + "px";
        tooltip.style.left = Math.max(8, rect.left - 100) + "px";

        const dismiss = (e) => {
            if (!tooltip.contains(e.target) && e.target !== anchor) {
                tooltip.remove();
                document.removeEventListener("click", dismiss);
            }
        };
        setTimeout(() => document.addEventListener("click", dismiss), 10);
    }

    // ── File upload ─────────────────────────────────────

    async function handleFiles(fileList) {
        if (!fileList || fileList.length === 0) return;

        for (const file of fileList) {
            await uploadFile(file);
        }
        loadDocuments();
    }

    async function uploadFile(file) {
        uploadProgress.classList.remove("hidden");
        progressFill.style.width = "0%";
        uploadStatus.textContent = `Caricamento ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            progressFill.style.width = "30%";

            const res = await fetch(`${API}/api/documents/upload`, {
                method: "POST",
                body: formData,
            });

            progressFill.style.width = "90%";

            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: "Errore upload" }));
                throw new Error(err.detail || err.error || "Upload fallito");
            }

            const data = await res.json();
            progressFill.style.width = "100%";
            uploadStatus.textContent = data.message || "Caricamento completato!";
            showToast(`📄 ${file.name}: ${data.chunks} frammenti indicizzati.`, "success");

        } catch (err) {
            uploadStatus.textContent = `Errore: ${err.message}`;
            showToast(`Errore caricamento ${file.name}: ${err.message}`, "error");
        }

        setTimeout(() => uploadProgress.classList.add("hidden"), 3000);
    }

    // ── Document list ───────────────────────────────────

    async function loadDocuments() {
        try {
            const res = await fetch(`${API}/api/documents`);
            if (!res.ok) return;

            const data = await res.json();

            if (!data.documents || data.documents.length === 0) {
                docList.innerHTML = '<p class="empty-state">Nessun documento caricato.</p>';
                return;
            }

            docList.innerHTML = "";
            data.documents.forEach((doc) => {
                const item = document.createElement("div");
                item.className = "doc-item";
                item.setAttribute("role", "listitem");
                item.innerHTML = `
                    <div class="doc-info">
                        <span class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
                        <span class="doc-meta">${doc.file_type.toUpperCase()} — ${doc.chunks} frammenti</span>
                    </div>
                    <button class="btn btn-icon doc-delete" aria-label="Elimina ${escapeHtml(doc.filename)}" title="Elimina">🗑️</button>
                `;

                item.querySelector(".doc-delete").addEventListener("click", async () => {
                    if (confirm(`Eliminare "${doc.filename}"?`)) {
                        await deleteDocument(doc.filename);
                    }
                });

                docList.appendChild(item);
            });
        } catch (err) {
            // Silently fail on startup if API not ready
        }
    }

    async function deleteDocument(filename) {
        try {
            const res = await fetch(`${API}/api/documents/${encodeURIComponent(filename)}`, {
                method: "DELETE",
            });
            if (res.ok) {
                showToast(`Documento "${filename}" eliminato.`, "success");
                loadDocuments();
            }
        } catch (err) {
            showToast("Errore durante l'eliminazione.", "error");
        }
    }

    // ── Toast notifications ─────────────────────────────

    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<div class="toast-text">${escapeHtml(message)}</div>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(50px)";
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    // ── Utilities ───────────────────────────────────────

    function scrollToBottom() {
        messages.scrollTop = messages.scrollHeight;
    }

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function formatText(text) {
        let html = escapeHtml(text);
        html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
        html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
        html = html.replace(/^- (.+)$/gm, "<li>$1</li>");
        html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => "<ul>" + m + "</ul>");
        html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
        html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
        html = html.replace(/\n\n/g, "</p><p>");
        html = "<p>" + html + "</p>";
        html = html.replace(/<p><\/p>/g, "");
        return html;
    }

    // ── Start ───────────────────────────────────────────

    document.addEventListener("DOMContentLoaded", init);
})();
