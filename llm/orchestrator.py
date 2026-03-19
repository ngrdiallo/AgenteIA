"""
LLM Orchestrator: sistema multi-backend con fallback a cascata e smart routing.

11-tier fallback con routing intelligente per task type:
1. Cerebras   (ultra-fast 1800 tok/s, Llama 3.3 70B — FREE)
2. Groq       (Llama 3.3 70B, ultra-fast — FREE tier)
3. Gemini     (2.5 Flash, 1M context window — FREE)
4. OpenRouter (modelli :free — DeepSeek R1, Kimi K2, Gemma 3n)
5. SambaNova  (Llama 3.3 70B, alta qualità — FREE tier)
6. Mistral    (mistral-small-latest — FREE tier)
7. HuggingFace Inference
8. DeepSeek
9. Ollama (local)
10. Risposta di errore esplicita

SMART ROUTING (basato su scouting report provider 2025):
- speed:         Cerebras → Groq     (bassa latenza)
- large_context: Gemini → SambaNova  (contesto enorme)
- reasoning:     OpenRouter R1:free → Groq
- general:       Cerebras → Groq → Gemini → OpenRouter → ...
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

from config import settings
from llm.backend_pool import BackendPool

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Risposta da un backend LLM."""
    text: str
    backend_used: str
    model: str = ""
    latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)



def _strip_think(text: str) -> str:
    """Remove <think> blocks from model outputs."""
    if not text:
        return text
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '')
    return text.strip()


class LLMOrchestrator:
    """
    Orchestratore multi-backend con fallback a cascata e smart routing.
    Prova i backend in ordine di priorità finché uno risponde.
    Riordina automaticamente in base a dimensione prompt e tipo di task.
    """

    # Ordine di fallback DEFAULT (ottimizzato per speed → quality)
    BACKEND_CHAIN = [
        "cerebras",     # 1800 tok/s, Llama 3.1 8B (FREE)
        "cloudflare",   # Edge inference, 10K neurons/day (FREE)
        "github",       # Llama-3.3-70B, 150 req/day (FREE)
        "groq",         # Ultra-fast, Llama 3.3 70B (FREE tier)
        "gemini",       # 2.5 Flash, 1M context (FREE)
        "nvidia",       # Llama-3.3-70B, 1000 credits (FREE)
        "aimlapi",      # 400+ modelli, 50K credits (FREE)
        "openrouter",   # Modelli :free (DeepSeek R1, Kimi K2, ...)
        "sambanova",    # Llama 3.3 70B (FREE tier)
        "mistral",      # Reliable fallback
        "ovh",          # EU provider, Llama 3.3 70B
        "scaleway",     # EU provider, Llama 3.3 70B
        "huggingface",
        "deepseek",
        "ollama",
    ]

    # Catene specializzate per tipo de task
    # VINCOLI LATENZA: speed ≤5s, focused ≤20s, comprehensive ≤45s
    _ROUTING_CHAINS = {
        # LARGE_CONTEXT: backbone Mistral + cloud reasoning + locale illimitato
        "large_context": [
            # TIER 1 — Volume alto, backbone
            "mistral",              # 33M token/mese
            "cerebras_qwen235",     # 1M/day, 1400 tok/s, 64K ctx FREE
            "gemini_lite",          # 1000 RPD (PRIMA di gemini!)
            "gemini",               # 20 RPD (dopo lite)

            # TIER 2 — Reasoning + qualità
            "glm5_cloud",           # Ollama cloud free preview
            "kimi_cloud",           # Ollama cloud free preview
            "openrouter_r1",        # DeepSeek R1 free
            "chutes",               # Bittensor decentralized
            "groq_gptoss",          # GPT-OSS-120B 1K/day

            # TIER 3 — Capacità alta, fallback primario
            "cloudflare",           # 10K req/day
            "modelscope_qwen",      # 2K req/day
            "nvidia",               # 40 RPM
            "cerebras_70b",         # 1M/day separato, 450 tok/s
            "cerebras_llama4",      # 1M/day, Llama4 Scout

            # TIER 4 — Fallback diversificato
            "groq_qwen3",           # Qwen3-32B Groq LPU
            "groq_kimi",            # Kimi-K2 Groq
            "groq_llama4",          # Llama4 Scout Groq
            "openrouter_llama4",    # Llama4 10M ctx
            "openrouter_qwen235",   # Qwen3-235B
            "sambanova", "openrouter_gemma",

            # TIER 5 — Ultimo cloud (deepseek removed - 402 permanent)
            "groq_8b", "groq",
            "openrouter_llama", "scaleway", "ovh",

            # TIER 6 — Locale illimitato
            "vllm", "llamacpp", "ollama",
        ],
        # Speed: Cerebras (illimitato) + Groq (veloce ma quota limitata)
        "speed": [
            "cerebras",             # llama3.1-8b, 1800 tok/s, sub-0.5s
            "cerebras_qwen32",      # Qwen3-32B, ~1400 tok/s
            "minimax_cloud",        # Ollama cloud fast
            "groq_8b",              # LPU sub-2s
            "groq",                 # LPU sub-3s
            "gemini_lite",          # 1000 RPD
            "cloudflare",           # edge, 10K/day
            "mistral",
            "llamacpp", "ollama",
        ],
        # Reasoning: modelli qualitativi
        "reasoning": [
            "cerebras_qwen235",     # Qwen3-235B 1400 tok/s reasoning ISTANTANEO
            "openrouter_r1",        # DeepSeek R1 SOTA
            "chutes",               # DeepSeek-R1 decentralized
            "groq_qwen3",           # Qwen3-32B /think LPU
            "glm5_cloud",           # GLM-5 744B
            "kimi_cloud",           # Kimi K2.5
            "modelscope_qwen",      # Qwen3-480B
            "mistral",
            "gemini",
            "groq_kimi", "nvidia", "cerebras", "groq", "sambanova",
        ],
    }

    # Provider permanentemente disabilitati (non verranno usati nelle chain)
    _DISABLED_PROVIDERS: set = {
        "deepseek",
        "cerebras_qwen235",       # 402 Insufficient Balance (permanente fino a ricarica manuale)
        "scaleway",       # API key non configurata
        "ovh",            # API key non configurata
        "aimlapi",        # API key non configurata
        "github",         # GitHub Copilot token non configurato
        "ollama_free_api",  # endpoint locale non attivo in produzione
        # Cloud providers without valid API keys - prevent them from going dead
        "glm5_cloud",     # Ollama cloud - needs valid endpoint
        "kimi_cloud",     # Ollama cloud - needs valid endpoint
        "modelscope_qwen",  # ModelScope - needs valid API key
        # Provider non necessari
        "nebius",         # Disabilitato su richiesta utente
        "cortecs",        # Disabilitato su richiesta utente
    }

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        preferred_backend: Optional[str] = None,
    ):
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.preferred_backend = preferred_backend or settings.LLM_DEFAULT_BACKEND
        self.last_response: Optional[LLMResponse] = None
        self.attempts_log: List[str] = []
        self.pool = BackendPool()

    # Limiti APPROSSIMATIVI di contesto per backend (token input)
    _BACKEND_INPUT_LIMITS = {
        "cerebras":      8000,     # Llama 3.1 8B: 8K context window
        "cerebras_70b": 128000,   # Llama 3.3 70B: 128K
        "cerebras_qwen235": 200000, # Qwen3-235B: 200K
        "cerebras_llama4": 10000000, # Llama4 Scout: 10M
        "cerebras_qwen32": 32000,  # Qwen3-32B: 32K
        "groq":          10000,    # Free tier: TPM limitato
        "groq_8b":       10000,    # 8B separate quota
        "groq_qwen3":    32000,    # Qwen3-32B: 32K context
        "groq_kimi":     32000,    # Kimi-K2: 32K context
        "groq_gptoss":   32000,    # GPT-OSS-120B: 32K context
        "groq_llama4":   10000000, # Llama4 Scout: 10M
        "gemini":        900000,   # Gemini 2.5 Flash: 1M context window
        "gemini_lite":   300000,   # 300K context
        "gemini_pro":    1000000,  # 1M context
        "openrouter":    60000,    # Modelli :free hanno 32-128K
        "openrouter_qwen235": 200000, # Qwen3-235B: 200K
        "sambanova":     60000,    # Llama 3.3 70B: 128K, free tier generoso
        "mistral":       120000,   # 128K context window
        "huggingface":   8000,     # 7B model: ~8K context
        "deepseek":      60000,    # 64K context
        "ollama":        8000,     # Dipende dal modello locale
        "modelscope_qwen": 128000, # Qwen3-Coder: 128K
        "chutes":        64000,    # DeepSeek R1: 64K
        "cloudflare":    32000,    # Workers AI: 32K context
        "nvidia":        32000,    # NIM: 32K context
        "fireworks":     32000,    # Fireworks AI: 32K context
        "vllm":          32768,    # vLLM default
        "llamacpp":      8192,     # llama.cpp default
        "localai":       8192,     # LocalAI default
    }

    # Timeout MASSIMO per singolo backend (secondi).
    # Impostati in modo che anche con 1-2 fallback si rispetti il budget tipo.
    _BACKEND_TIMEOUTS = {
        "cerebras":       8,     # Ultra-fast: se non risponde in 8s, è morto
        "cerebras_70b":  15,     # 70B larger but still fast
        "cerebras_qwen235": 12,  # Qwen3-235B fast
        "cerebras_llama4": 15,   # Llama4 Scout
        "cerebras_qwen32": 10,   # Qwen3-32B fast
        "groq":           8,      # Ultra-fast
        "groq_8b":        8,      # 8B models are faster
        "groq_qwen3":    15,     # Qwen3-32B reasoning
        "groq_kimi":     15,     # Kimi-K2
        "groq_gptoss":   20,     # GPT-OSS-120B larger model
        "groq_llama4":   15,     # Llama4 Scout
        "gemini":        50,     # Large context: più tempo
        "gemini_lite":   15,     # Flash-Lite: veloce
        "gemini_pro":    60,     # Pro: slower but higher quality
        "sambanova":     35,     # Large context but has limits
        "mistral":       45,     # Cap 45s — fallback ha 15s residui
        "huggingface":   15,     # Medio
        "deepseek":      45,     # Buona capacità
        "openrouter":    45,     # Dipende dal modello
        "openrouter_r1": 50,     # DeepSeek R1 reasoning
        "openrouter_llama": 45,
        "openrouter_llama4": 60,
        "openrouter_qwen": 45,
        "openrouter_qwen235": 55, # Qwen3-235B
        "openrouter_gemma": 45,
        "ovh":           20,     # EU provider
        "scaleway":      20,     # EU provider
        "ollama":        30,     # Locale
        "ollama_free_api": 45,   # Distributed
        "github":        20,     # Stabile, 70B
        "nvidia":        40,     # Stabile, 70B
        "cloudflare":    15,     # Edge inference, bassa latenza
        "aimlapi":       20,     # 70B Turbo
        "fireworks":     35,     # Fast, good capacity
        # Ollama Cloud models - via localhost:11434
        "glm5_cloud":    60,     # 744B params, reasoning heavy
        "kimi_cloud":    50,     # multimodal
        "minimax_cloud": 40,     # fast agentic
        # New providers
        "modelscope_qwen": 60,   # Qwen3-480B, 2K req/day
        "chutes":        45,     # Bittensor decentralized
        "vllm":          30,     # Local GPU
        "llamacpp":      60,     # Local CPU
        "localai":       60,     # Local full stack
    }

    # Budget TOTALE per tipo di query (secondi). Se superato → stop tentativi.
    # NOTA: questo è il budget per la sola generazione LLM.
    # Il pipeline E2E (retrieval + context + LLM) consuma ~5-10s in più.
    # I limiti E2E sono: speed=5s, reasoning=20s, large_context=45s.
    # Quindi i budget LLM sono ridotti per lasciare margine al pipeline.
    _QUERY_TYPE_BUDGETS = {
        "speed":         4,    # meta, greeting (E2E target: 5s)
        "reasoning":     17,   # focused (E2E target: 20s)
        "large_context": 90,   # comprehensive: Mistral 45s + fallback 45s
        None:            20,   # default
    }

    def _estimate_tokens(self, text: str) -> int:
        """Stima approssimativa dei token (1 token ≈ 4 caratteri per l'italiano)."""
        return max(1, len(text) // 4)

    def _call_with_timeout(
        self, method: Callable, prompt: str, system_prompt: str, timeout_s: int
    ) -> Tuple[str, bool, Dict]:
        """
        Esegue un backend con timeout rigido.
        Se il backend non risponde entro timeout_s secondi → TimeoutError.
        Questo GARANTISCE latenza massima prevedibile per l'utente.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(method, prompt, system_prompt)
        try:
            result = future.result(timeout=timeout_s)
            executor.shutdown(wait=False)
            return result
        except concurrent.futures.TimeoutError:
            # Non aspettare il thread zombie — lascialo morire da solo
            executor.shutdown(wait=False)
            raise

    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        on_attempt: Optional[Callable[[str, str], None]] = None,
        routing_hint: Optional[str] = None,
    ) -> LLMResponse:
        """
        Genera una completion usando la catena di fallback con smart routing.

        Rispetta un budget TOTALE per tipo di query:
        - speed (meta/greeting): 5s
        - reasoning (focused):   20s
        - large_context (comp):  45s

        Se il budget è esaurito, smette di provare backend e ritorna errore.
        """
        self.attempts_log = []
        total_start = time.time()
        total_budget = self._QUERY_TYPE_BUDGETS.get(routing_hint, 30)

        estimated_input_tokens = self._estimate_tokens(prompt + system_prompt)

        # Se un backend specifico è richiesto, prova solo quello
        if self.preferred_backend and self.preferred_backend != "auto":
            chain = [self.preferred_backend]
        else:
            # Smart routing via BackendPool: esclude automaticamente
            # provider in cooldown / con contesto insufficiente
            effective_hint = routing_hint
            if not effective_hint and estimated_input_tokens > 15000:
                effective_hint = "large_context"
            chain = self.pool.get_chain(
                effective_hint or "general", estimated_input_tokens
            )
            # Rimuovi provider disabilitati dalla chain
            chain = [b for b in chain if b not in self._DISABLED_PROVIDERS]
            logger.info(
                f"🏊 Pool chain ({effective_hint or 'general'}): {chain[:5]}"
            )

        last_error = ""
        for backend_name in chain:
            # Controlla budget totale PRIMA di provare il prossimo backend
            elapsed = time.time() - total_start
            remaining = total_budget - elapsed
            if remaining < 1.0:
                self.attempts_log.append(
                    f"⏱️ Budget esaurito ({total_budget}s) dopo {elapsed:.1f}s"
                )
                logger.warning(
                    f"Budget tipo {routing_hint} esaurito ({total_budget}s). "
                    f"Elapsed: {elapsed:.1f}s. Stop tentativi."
                )
                break
            # Nota: il filtro per contesto insufficiente è già in pool.get_chain()
            method = self._get_backend_method(backend_name)
            if method is None:
                continue

            # Skip provider disabilitati permanentemente
            if backend_name in self._DISABLED_PROVIDERS:
                logger.debug(f"⛔ Skip {backend_name}: disabilitato")
                continue

            if on_attempt:
                on_attempt(backend_name, "trying")

            # Timeout = min(backend timeout, remaining budget)
            backend_timeout = self._BACKEND_TIMEOUTS.get(backend_name, 15)
            elapsed = time.time() - total_start
            remaining = total_budget - elapsed
            timeout_s = max(2, min(backend_timeout, int(remaining)))
            try:
                text, success, metadata = self._call_with_timeout(
                    method, prompt, system_prompt, timeout_s
                )
                if success and text.strip():
                    latency_ms = metadata.get("latency_s", 0.0) * 1000
                    self.pool.report_success(backend_name, latency_ms)
                    self.attempts_log.append(f"✅ {backend_name}")
                    response = LLMResponse(
                        text=text.strip(),
                        backend_used=backend_name,
                        model=metadata.get("model", ""),
                        latency=metadata.get("latency_s", 0.0),
                        metadata=metadata,
                    )
                    self.last_response = response
                    if on_attempt:
                        on_attempt(backend_name, "success")
                    return response
                else:
                    self.pool.report_failure(backend_name, "empty_response")
                    self.attempts_log.append(f"⚠️ {backend_name}: risposta vuota")

            except (concurrent.futures.TimeoutError, TimeoutError):
                self.pool.report_failure(backend_name, "timeout")
                self.attempts_log.append(
                    f"⏱️ {backend_name}: timeout ({timeout_s}s)"
                )
                logger.warning(
                    f"Backend {backend_name}: timeout dopo {timeout_s}s, fallback..."
                )
                if on_attempt:
                    on_attempt(backend_name, f"timeout: {timeout_s}s")

            except Exception as e:
                error_str = str(e)[:200]
                error_lower = error_str.lower()

                # Classificazione errore MUTUAMENTE ESCLUSIVA
                _RATE_LIMIT_KW = [
                    "429", "rate limit", "rate_limit", "too many requests",
                    "quota exceeded", "resource_exhausted", "queue_exceeded",
                    "tokensperminute", "requestsperminute", "daily limit", "daily quota", "per day", "rate_limit_exceeded",
                    "rate_limit_from_provider", "rate_limit_provider",
                    # Cloudflare Workers AI daily cap
                    "3036", "daily free allocation", "neurons",
                ]
                _PAYMENT_KW = [
                    "402",
                    "insufficient balance",       # DeepSeek
                    "requires more credits",      # OpenRouter
                    "can only afford",            # OpenRouter max_tokens > credits
                    "payment required",           # Fireworks, generico
                    "exceeded your monthly",      # Fireworks, HuggingFace
                    "insufficient credits",       # OpenRouter alternativo
                    "upgrade your plan",          # Fireworks/HuggingFace
                    "billing",                    # generico pay-wall
                ]
                is_429 = any(kw in error_lower for kw in _RATE_LIMIT_KW)
                is_402 = any(kw in error_lower for kw in _PAYMENT_KW)
                is_context = "context_length" in error_lower or "too long" in error_lower

                if is_429:
                    # Provider sano, quota temporanea esaurita.
                    # Calcola cooldown preciso da Groq "try again in Xm Ys" o "in Xs" o Gemini "retry_delay { seconds: N }"
                    import re as _re
                    cooldown = 65  # default sicuro
                    m_min   = _re.search(r'(\d+)m([\d.]+)s', error_str)
                    m_proto = _re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', error_str)  # Gemini proto
                    m_sec   = _re.search(r'in ([\d.]+)s', error_str)
                    if m_min:
                        cooldown = int(m_min.group(1)) * 60 + float(m_min.group(2)) + 2
                    elif m_proto:
                        cooldown = float(m_proto.group(1)) + 2
                    elif m_sec:
                        cooldown = float(m_sec.group(1)) + 2
                    is_openrouter = backend_name.startswith("openrouter")
                    is_daily_cap = any(kw in error_lower for kw in ["daily limit", "daily quota", "per day", "rate_limit_exceeded"])
                    if is_openrouter and is_daily_cap:
                        cooldown = 86400
                    else:
                        cooldown = max(10, min(int(cooldown), 300))  # clamp [10, 300]s
                    self.pool.record_rate_limit(backend_name, cooldown_seconds=cooldown, error_msg=error_str)
                    logger.info(f"⏳ {backend_name}: rate limited, cooldown {cooldown}s")
                elif is_402:
                    # Balance esaurito — PERMANENTE, non si auto-risolve.
                    # Disabilita il provider per 24h.
                    self.pool.record_rate_limit(backend_name, cooldown_seconds=86400, error_msg=error_str)
                    logger.warning(f"💸 {backend_name}: insufficient balance (402), disabled 24h")
                elif is_context:
                    # Il prompt è troppo grande per questo provider.
                    # Non è un guasto del provider.
                    self.pool.report_failure(backend_name, "context_skip", is_context_error=True)
                    logger.debug(f"📏 {backend_name}: context too long, skip (no penalty)")
                else:
                    # Errore strutturale reale: connessione rotta, 500, timeout duro.
                    self.pool.report_failure(backend_name, "generic")

                last_error = f"{backend_name}: {error_str[:100]}"
                self.attempts_log.append(f"❌ {backend_name}: {str(e)[:60]}")
                logger.warning(f"Backend {backend_name} fallito: {e}")
                if on_attempt:
                    on_attempt(backend_name, f"failed: {str(e)[:50]}")
                continue

        # Tutti i backend falliti - ritorna errore
        logger.error(f"Tutti i backend LLM falliti. Ultimo errore: {last_error}")
        
        return LLMResponse(
            text="Mi dispiace, al momento non riesco a elaborare la richiesta. "
                 "Riprova tra qualche istante.",
            backend_used="none",
            metadata={"error": last_error, "attempts": self.attempts_log},
        )

    def _is_rate_limit_error(self, error_info: dict) -> bool:
        """
        Rileva se l'errore è un 429 rate limit.
        """
        err_str = str(error_info.get("error", "")).lower()
        return any(kw in err_str for kw in [
            "429", "rate limit", "rate_limit", "too many requests",
            "quota exceeded", "resource_exhausted", "queue_exceeded",
            "tokensperminute", "requestsperminute", "daily limit",
        ])

    def _get_backend_method(self, name: str) -> Optional[Callable]:
        """Mappa nome backend → metodo."""
        methods = {
            "cerebras": self._query_cerebras,
            "openrouter": self._query_openrouter,
            "openrouter_r1": self._query_openrouter_r1,
            "openrouter_llama": self._query_openrouter_llama,
            "openrouter_llama4": self._query_openrouter_llama4,
            "openrouter_qwen": self._query_openrouter_qwen,
            "openrouter_gemma": self._query_openrouter_gemma,
            "groq": self._query_groq,
            "groq_8b": self._query_groq_8b,
            "mistral": self._query_mistral,
            "gemini": self._query_gemini,
            "gemini_lite": self._query_gemini_lite,
            "sambanova": self._query_sambanova,
            "huggingface": self._query_huggingface,
            "deepseek": self._query_deepseek,
            "ovh": self._query_ovh,
            "scaleway": self._query_scaleway,
            "ollama": self._query_ollama,
            "ollama_free_api": self._query_ollama_free_api,
            "glm5_cloud": self._query_glm5_cloud,
            "kimi_cloud": self._query_kimi_cloud,
            "minimax_cloud": self._query_minimax_cloud,
            "github": self._query_github,
            "nvidia": self._query_nvidia,
            "cloudflare": self._query_cloudflare,
            "aimlapi": self._query_aimlapi,
            "fireworks": self._query_fireworks,
            # New Groq models with separate quota
            "groq_qwen3": self._query_groq_qwen3,
            "groq_kimi": self._query_groq_kimi,
            "groq_gptoss": self._query_groq_gptoss,
            "groq_llama4": self._query_groq_llama4,
            # ModelScope
            "modelscope_qwen": self._query_modelscope_qwen,
            # Chutes.ai - decentralized Bittensor
            "chutes": self._query_chutes,
            # IO.NET - 500K token/giorno gratis
            "ionet": self._query_ionet,
            # Local providers
            "vllm": self._query_vllm,
            "llamacpp": self._query_llamacpp,
            "localai": self._query_localai,
            # OpenRouter sub-providers
            "openrouter_qwen235": self._query_openrouter_qwen235,
            # Cerebras multi-model with separate quota
            "cerebras_70b": self._query_cerebras_70b,
            "cerebras_qwen235": self._query_cerebras_qwen235,
            "cerebras_llama4": self._query_cerebras_llama4,
            "cerebras_qwen32": self._query_cerebras_qwen32,
            # Gemini Pro
            "gemini_pro": self._query_gemini_pro,
        }
        return methods.get(name)

    # ------------------------------------------------------------------
    # Backend: Cerebras (OpenAI-compatible, ultra-fast — FREE)
    # Multi-model fallback: se un modello è 404, prova il successivo
    # ------------------------------------------------------------------
    CEREBRAS_MODELS = [
        "llama3.1-8b",       # veloce, sempre disponibile
        "gpt-oss-120b",      # più grande quando disponibile
    ]

    def _query_cerebras(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.CEREBRAS_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for model in self.CEREBRAS_MODELS:
            try:
                client = openai.OpenAI(
                    base_url="https://api.cerebras.ai/v1",
                    api_key=api_key,
                )
                start = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = time.time() - start
                text = response.choices[0].message.content or ""
                return text, bool(text.strip()), {
                    "model": model,
                    "provider": "Cerebras",
                    "latency_s": round(latency, 2),
                }
            except Exception as e:
                err = str(e)
                if "404" in err or "model" in err.lower():
                    last_error = e
                    logger.debug(f"Cerebras model {model} non disponibile, provo il prossimo")
                    continue
                raise
        if last_error:
            raise last_error
        return "", False, {}

    def _query_cerebras_model(self, prompt: str, system_prompt: str, model: str) -> Tuple[str, bool, Dict]:
        """Cerebras base method with model override for separate quota."""
        api_key = settings.CEREBRAS_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            # Limit max_tokens to avoid pre-rate-limit estimation
            cerebras_max_tokens = min(self.max_tokens, 2048)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=cerebras_max_tokens,
                timeout=30,
            )
            latency = time.time() - start
            text = response.choices[0].message.content or ""
            return _strip_think(text), bool(text.strip()), {
                "model": model,
                "provider": "Cerebras-WSE",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e), "model": model}

    def _query_cerebras_70b(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Cerebras Llama 3.3 70B — 1M token/day separate quota, 450 tok/s."""
        return self._query_cerebras_model(prompt, system_prompt, "llama3.3-70b")

    def _query_cerebras_qwen235(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Cerebras Qwen3-235B — 1M token/day separate quota, 1400 tok/s."""
        return self._query_cerebras_model(prompt, system_prompt, "qwen-3-235b-a22b")

    def _query_cerebras_llama4(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Cerebras Llama 4 Scout — 1M token/day separate quota."""
        return self._query_cerebras_model(prompt, system_prompt, "llama-4-scout")

    def _query_cerebras_qwen32(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Cerebras Qwen3-32B — 1M token/day separate quota, 1400 tok/s."""
        return self._query_cerebras_model(prompt, system_prompt, "qwen3-32b")

    # ------------------------------------------------------------------
    # Backend: OpenRouter (modelli :free — DeepSeek R1, Kimi K2, Gemma 3n)
    # Rotazione automatica su modelli gratuiti con fallback
    # ------------------------------------------------------------------
    _OPENROUTER_FREE_MODELS = [
        "deepseek/deepseek-r1-0528:free",
        "moonshotai/kimi-k2:free",
        "google/gemma-3n-e4b-it:free",
        "nousresearch/deephermes-3-llama-3-8b:free",
    ]
    def _query_openrouter(self, prompt: str, system_prompt: str, model: str = None) -> Tuple[str, bool, Dict]:
        """OpenRouter multi-model. Se model è specificato usa fast-path diretto."""
        import openai as _oai
        import time as _t

        api_key = settings.OPENROUTER_API_KEY
        if not api_key:
            return "", False, {"error": "OPENROUTER_API_KEY non configurata"}

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        def _is_rate_limit_error(err: str) -> bool:
            err_lower = err.lower()
            return any(kw in err_lower for kw in [
                "429", "rate limit", "rate_limit", "too many requests",
                "quota exceeded", "resource_exhausted",
                "daily limit", "daily quota", "per day",
                "rate_limit_from_provider", "rate_limit_provider",
            ])

        if model:
            try:
                client = _oai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                start = _t.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = _t.time() - start
                text = _strip_think(response.choices[0].message.content or "")
                return text, bool(text.strip()), {
                    "model": model,
                    "provider": "OpenRouter",
                    "latency_s": round(latency, 2),
                }
            except Exception as e:
                err = str(e)
                if _is_rate_limit_error(err):
                    raise Exception(f"429 rate_limit_from_provider: {err[:180]}")
                logger.warning(f"openrouter direct model={model} err: {err[:120]}")
                return "", False, {"error": err, "model": model}

            client = _oai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        last_err = ""
        saw_rate_limit = False
        last_rate_limit_err = ""
        for model_name in self._OPENROUTER_FREE_MODELS:
            try:
                start = _t.time()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = _t.time() - start

                text = response.choices[0].message.content or ""
                if text.strip():
                    model_used = getattr(response, "model", model_name)
                    return _strip_think(text), True, {
                        "model": model_used,
                        "provider": "OpenRouter",
                        "latency_s": round(latency, 2),
                    }
            except Exception as e:
                err = str(e)
                last_err = err[:120]
                if _is_rate_limit_error(err):
                    saw_rate_limit = True
                    last_rate_limit_err = err
                logger.debug(f"OpenRouter {model_name} fallito: {last_err}")
                continue

        if saw_rate_limit:
            raise Exception(f"429 rate_limit_from_provider: {(last_rate_limit_err or last_err)[:180]}")

        logger.warning(f"OpenRouter: tutti i modelli :free falliti. Ultimo: {last_err}")
        return "", False, {"error": last_err}

    # ------------------------------------------------------------------
    # Backend: Groq — multi-model fallback
    # groq_8b has separate quota from groq_70b
    # ------------------------------------------------------------------
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",  # Main model
        "llama-3.1-8b-instant",     # Separate quota: 100K token/day
    ]

    def _query_groq(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.GROQ_API_KEY
        if not api_key:
            return "", False, {}

        from groq import Groq

        client = Groq(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for model in self.GROQ_MODELS:
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                latency = time.time() - start
                text = response.choices[0].message.content or ""
                return text, bool(text.strip()), {
                    "model": model,
                    "provider": "Groq",
                    "latency_s": round(latency, 2),
                }
            except Exception as e:
                err = str(e)
                if "404" in err or "model" in err.lower() or "not found" in err.lower():
                    last_error = e
                    logger.debug(f"Groq model {model} non disponibile, provo il prossimo")
                    continue
                raise
        if last_error:
            raise last_error
        return "", False, {}

    def _query_groq_model(self, prompt: str, system_prompt: str, model: str, reasoning_format: str = None) -> Tuple[str, bool, Dict]:
        """Groq helper method for models with separate quota."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            return "", False, {}

        from groq import Groq

        client = Groq(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            # Only pass reasoning_format for reasoning models
            create_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if reasoning_format:
                create_kwargs["reasoning_format"] = reasoning_format
            
            response = client.chat.completions.create(**create_kwargs)
            latency = time.time() - start
            text = response.choices[0].message.content or ""
            return text, bool(text.strip()), {
                "model": model,
                "provider": "Groq",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e), "model": model}

    # ------------------------------------------------------------------
    # Backend: Groq 8B — llama-3.1-8b-instant with SEPARATE QUOTA
    # ------------------------------------------------------------------
    def _query_groq_8b(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Groq llama-3.1-8b-instant — quota completely separate from 70B model."""
        api_key = settings.GROQ_API_KEY
        if not api_key:
            return "", False, {}

        from groq import Groq

        client = Groq(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            latency = time.time() - start
            text = response.choices[0].message.content or ""
            return text, bool(text.strip()), {
                "model": "llama-3.1-8b-instant",
                "provider": "Groq-8B",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            logger.warning(f"Groq-8B failed: {e}")
            return "", False, {"error": str(e)}

    # ------------------------------------------------------------------
    # Backend: Mistral
    # PORTED FROM: advanced_reasoning_llm.py _query_mistral
    # ------------------------------------------------------------------
    def _query_mistral(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.MISTRAL_API_KEY
        if not api_key:
            return "", False, {}

        from mistralai import Mistral

        client = Mistral(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "mistral-small-latest",
            "provider": "Mistral",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: Google Gemini 2.5 Flash (1M context, FREE tier)
    # Upgrade da 2.0-flash-lite → 2.5-flash
    # ------------------------------------------------------------------
    def _query_gemini(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            return "", False, {}

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=system_prompt if system_prompt else None,
        )

        start = time.time()
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        latency = time.time() - start

        text = response.text or ""
        return text, bool(text.strip()), {
            "model": "gemini-2.5-flash",
            "provider": "Google",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: HuggingFace Inference
    # PORTED FROM: advanced_reasoning_llm.py _query_huggingface
    # ------------------------------------------------------------------
    def _query_huggingface(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        hf_token = settings.HF_TOKEN
        if not hf_token:
            return "", False, {}

        from huggingface_hub import InferenceClient

        client = InferenceClient(api_key=hf_token)

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        start = time.time()
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider="hf-inference",
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "Mistral-7B-Instruct-v0.3",
            "provider": "HuggingFace",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: SambaNova (Llama 3.3 70B — FREE tier, 10-30 RPM)
    # API: https://api.sambanova.ai/v1 — cloud.sambanova.ai
    # ------------------------------------------------------------------
    def _query_sambanova(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.SAMBANOVA_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://api.sambanova.ai/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "Meta-Llama-3.3-70B-Instruct",
            "provider": "SambaNova",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: DeepSeek
    # ------------------------------------------------------------------
    def _query_deepseek(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "deepseek-chat",
            "provider": "DeepSeek",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: OVH AI Endpoints (Meta-Llama-3.3-70B-Instruct — EU)
    # API: https://endpoints.ai.cloud.ovh.net/v1
    # ------------------------------------------------------------------
    def _query_ovh(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.OVH_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://endpoints.ai.cloud.ovh.net/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = client.chat.completions.create(
            model="Meta-Llama-3.3-70B-Instruct",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "Meta-Llama-3.3-70B-Instruct",
            "provider": "OVH",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: Scaleway (llama-3.3-70b-instruct — EU)
    # API: https://api.scaleway.ai/v1
    # ------------------------------------------------------------------
    def _query_scaleway(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.SCALEWAY_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://api.scaleway.ai/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        response = client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.time() - start

        text = response.choices[0].message.content or ""
        return text, bool(text.strip()), {
            "model": "llama-3.3-70b-instruct",
            "provider": "Scaleway",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: GitHub Models (Llama-3.3-70B gratis — 150 req/day "Low")
    # API: https://models.inference.ai.azure.com — OpenAI-compatible
    # Multi-model fallback interno
    # ------------------------------------------------------------------
    GITHUB_MODELS = [
        "Llama-3.3-70B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "Mistral-small",
    ]

    def _query_github(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        token = settings.GITHUB_TOKEN
        if not token:
            return "", False, {}

        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for model in self.GITHUB_MODELS:
            try:
                start = time.time()
                with httpx.Client(timeout=20.0) as client:
                    r = client.post(
                        "https://models.inference.ai.azure.com/chat/completions",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "messages": messages,
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                        },
                    )
                    r.raise_for_status()
                latency = time.time() - start
                text = r.json()["choices"][0]["message"]["content"] or ""
                return text, bool(text.strip()), {
                    "model": model,
                    "provider": "GitHub Models",
                    "latency_s": round(latency, 2),
                }
            except Exception as e:
                err = str(e)
                if "404" in err or "model" in err.lower():
                    last_error = e
                    logger.debug(f"GitHub model {model} non disponibile, provo il prossimo")
                    continue
                raise
        if last_error:
            raise last_error
        return "", False, {}

    # ------------------------------------------------------------------
    # Backend: NVIDIA NIM (Llama-3.3-70B — 1000 credits gratis)
    # API: https://integrate.api.nvidia.com/v1 — OpenAI-compatible
    # ------------------------------------------------------------------
    def _query_nvidia(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.NVIDIA_API_KEY
        if not api_key:
            return "", False, {}

        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        with httpx.Client(timeout=20.0) as client:
            r = client.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
            )
            r.raise_for_status()
        latency = time.time() - start

        text = r.json()["choices"][0]["message"]["content"] or ""
        return text, bool(text.strip()), {
            "model": "meta/llama-3.3-70b-instruct",
            "provider": "NVIDIA NIM",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: Cloudflare Workers AI (edge inference — 10K neurons/day)
    # API: cf/meta/llama-3.1-8b-instruct — ultra-bassa latenza
    # ------------------------------------------------------------------
    def _query_cloudflare(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        token = settings.CLOUDFLARE_API_TOKEN
        account_id = settings.CLOUDFLARE_ACCOUNT_ID
        if not token or not account_id:
            return "", False, {}

        import httpx

        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
            )
            r.raise_for_status()
        latency = time.time() - start

        data = r.json()
        text = data.get("result", {}).get("response", "") or ""
        return text, bool(text.strip()), {
            "model": "@cf/meta/llama-3.1-8b-instruct",
            "provider": "Cloudflare",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: AIMLAPI (400+ modelli — OpenAI-compatible, 50K credits)
    # API: https://api.aimlapi.com/v1
    # ------------------------------------------------------------------
    def _query_aimlapi(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.AIMLAPI_KEY
        if not api_key:
            return "", False, {}

        import httpx

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        with httpx.Client(timeout=20.0) as client:
            r = client.post(
                "https://api.aimlapi.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            r.raise_for_status()
        latency = time.time() - start

        text = r.json()["choices"][0]["message"]["content"] or ""
        return text, bool(text.strip()), {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "provider": "AIMLAPI",
            "latency_s": round(latency, 2),
        }

    # ------------------------------------------------------------------
    # Backend: Ollama (locale)
    # PORTED FROM: advanced_reasoning_llm.py _query_ollama
    # ------------------------------------------------------------------
    def _query_ollama(self, prompt: str, system_prompt: str, model: str = None) -> Tuple[str, bool, Dict]:
        import requests

        base_url = "http://localhost:11434"

        # Verifica che Ollama sia in esecuzione
        try:
            requests.get(f"{base_url}/api/tags", timeout=2)
        except Exception:
            return "", False, {}

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        selected_model = model or "llama3.2"

        start = time.time()
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": selected_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            timeout=120,
        )
        latency = time.time() - start

        if response.status_code == 200:
            data = response.json()
            text = data.get("response", "")
            return text, bool(text.strip()), {
                "model": data.get("model", selected_model),
                "provider": "Ollama",
                "latency_s": round(latency, 2),
            }

        return "", False, {}

    # ------------------------------------------------------------------
    # Backend: Gemini Flash-Lite (gemini-2.0-flash-lite — 1000 RPD, ultra-fast)
    # ------------------------------------------------------------------
    def _query_gemini_lite(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Gemini 2.0 Flash via OpenAI-compatible endpoint."""
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "", False, {"error": "GOOGLE_API_KEY non configurata"}
        return self._query_openai_compatible(
            prompt=prompt,
            system_prompt=system_prompt,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model="gemini-2.0-flash",
            timeout=60,
        )

        genai.configure(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "user", "parts": [system_prompt]})
        messages.append({"role": "user", "parts": [prompt]})

        try:
            start = time.time()
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            response = model.generate_content(
                messages,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            latency = time.time() - start

            text = response.text or ""
            return text, bool(text.strip()), {
                "model": "gemini-2.0-flash-lite",
                "provider": "google",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e)}

    # ------------------------------------------------------------------
    # Backend: Ollama Cloud GLM-5 (744B params, reasoning SOTA, via localhost:11434)
    # ------------------------------------------------------------------
    def _query_glm5_cloud(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """GLM-5 Cloud — 744B params, reasoning SOTA, via Ollama cloud."""
        return self._query_ollama(prompt, system_prompt, model="glm-5:cloud")

    # ------------------------------------------------------------------
    # Backend: Ollama Cloud Kimi-K2.5 (multimodal agentic, via localhost:11434)
    # ------------------------------------------------------------------
    def _query_kimi_cloud(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Kimi-K2.5 Cloud — multimodal agentic, via Ollama cloud."""
        return self._query_ollama(prompt, system_prompt, model="kimi-k2.5:cloud")

    # ------------------------------------------------------------------
    # Backend: Ollama Cloud MiniMax-M2.5 (fast agentic, via localhost:11434)
    # ------------------------------------------------------------------
    def _query_minimax_cloud(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """MiniMax-M2.5 Cloud — fast agentic workflows, via Ollama cloud."""
        return self._query_ollama(prompt, system_prompt, model="minimax-m2.5:cloud")

    # ------------------------------------------------------------------
    # Backend: OllamaFreeAPI (distributed, no API key required)
    # ------------------------------------------------------------------
    def _query_ollama_free_api(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """OllamaFreeAPI — 50+ modelli, no API key, load-balanced globale."""
        import openai

        client = openai.OpenAI(
            base_url="https://ollamafreeapi.com/v1",
            api_key="no-key-required"
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            resp = client.chat.completions.create(
                model="llama3.3:70b",
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=45
            )
            latency = time.time() - start

            text = resp.choices[0].message.content or ""
            return text, bool(text.strip()), {
                "model": "llama3.3:70b",
                "provider": "OllamaFreeAPI",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e)}

    # ------------------------------------------------------------------
    # Backend: OpenRouter DeepSeek R1 (reasoning SOTA)
    # ------------------------------------------------------------------
    def _query_openrouter_r1(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        return self._query_openrouter(prompt, system_prompt, model="deepseek-r1:free")

    # ------------------------------------------------------------------
    # Backend: OpenRouter Llama 3.3 70B
    # ------------------------------------------------------------------
    def _query_openrouter_llama(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        return self._query_openrouter(prompt, system_prompt, model="meta-llama/llama-3.3-70b-instruct:free")

    # ------------------------------------------------------------------
    # Backend: OpenRouter Llama 4 Scout (10M context!)
    # ------------------------------------------------------------------
    def _query_openrouter_llama4(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        return self._query_openrouter(prompt, system_prompt, model="meta-llama/llama-4-scout:free")

    # ------------------------------------------------------------------
    # Backend: OpenRouter Qwen 2.5 72B
    # ------------------------------------------------------------------
    def _query_openrouter_qwen(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        return self._query_openrouter(prompt, system_prompt, model="qwen/qwen-2.5-72b-instruct:free")

    # ------------------------------------------------------------------
    # Backend: OpenRouter Gemma 3 27B
    # ------------------------------------------------------------------
    def _query_openrouter_gemma(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        return self._query_openrouter(prompt, system_prompt, model="google/gemma-3-27b-it:free")

    # ------------------------------------------------------------------
    # Backend: Fireworks AI (Llama 3.3 70B, fast, 131K context)
    # ------------------------------------------------------------------
    def _query_fireworks(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        api_key = settings.FIREWORKS_API_KEY
        if not api_key:
            return "", False, {}

        import openai

        client = openai.OpenAI(
            base_url="https://api.fireworks.ai/v1",
            api_key=api_key,
        )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p3-70b-instruct",
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=35,
            )
            latency = time.time() - start

            text = response.choices[0].message.content or ""
            return text, bool(text.strip()), {
                "model": "llama-v3p3-70b-instruct",
                "provider": "fireworks",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e)}

    # ==============================================================================================
    # SHARED METHOD: OpenAI-Compatible API base per ridurre duplicazione
    # ==============================================================================================
    def _query_openai_compatible(
        self,
        prompt: str,
        system_prompt: str,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 30,
    ) -> Tuple[str, bool, Dict]:
        """Base method for all OpenAI-compatible endpoints (ModelScope, vLLM, llama.cpp, LocalAI)."""
        import openai

        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout,
            )
            latency = time.time() - start

            text = response.choices[0].message.content or ""
            return text, bool(text.strip()), {
                "model": model,
                "provider": base_url.split("//")[1].split("/")[0] if "//" in base_url else base_url,
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            err_str = str(e)
            # Riconosci 429, quota exceeded, rate limit e simili
            # e propaga l'errore con marker "429" per il main dispatch
            err_lower = err_str.lower()
            if ("429" in err_str or "rate_limit" in err_lower or 
                "too many requests" in err_lower or "quota" in err_lower or
                "resource_exhausted" in err_lower):
                # Ri-solleva con marker 429 che l'orchestrator riconosce
                raise Exception(f"429 rate_limit_from_provider: {err_str[:150]}")
            return "", False, {"error": err_str}

    # ------------------------------------------------------------------
    # NEW Groq Models with SEPARATE QUOTA
    # ------------------------------------------------------------------
    def _query_groq_qwen3(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Groq Qwen3-32B — reasoning with /think mode, 1K req/day separate quota."""
        return self._query_groq_model(prompt, system_prompt, "qwen/qwen3-32b", reasoning_format="hidden")

    def _query_groq_kimi(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Groq Kimi-K2 — multimodal reasoning, 1K req/day separate quota."""
        return self._query_groq_model(prompt, system_prompt, "moonshotai/kimi-k2-instruct-0905", reasoning_format="hidden")

    def _query_groq_gptoss(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Groq GPT-OSS-120B — OpenAI open source, 1K req/day separate quota."""
        return self._query_groq_model(prompt, system_prompt, "openai/gpt-oss-120b")

    # ------------------------------------------------------------------
    # ModelScope Qwen3-Coder-480B — 2K req/day FREE
    # ------------------------------------------------------------------
    def _query_modelscope_qwen(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """ModelScope Qwen3-Coder-480B — 2K req/day FREE, 480B params."""
        api_key = settings.MODELSCOPE_API_KEY
        if not api_key:
            return "", False, {"error": "no modelscope key"}

        return self._query_openai_compatible(
            prompt, system_prompt,
            base_url="https://api-inference.modelscope.cn/v1/",
            api_key=api_key,
            model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
            timeout=60,
        )

    # ------------------------------------------------------------------
    # Local Providers: vLLM, llama.cpp, LocalAI
    # ------------------------------------------------------------------
    def _query_vllm(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """vLLM — GPU optimized, 2-4x faster than llama.cpp with GPU."""
        return self._query_openai_compatible(
            prompt, system_prompt,
            base_url="http://localhost:8001/v1",
            api_key="none",
            model="qwen3-8b",
            timeout=30,
        )

    def _query_llamacpp(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """llama.cpp server — ultra-light, runs on any hardware, unlimited."""
        return self._query_openai_compatible(
            prompt, system_prompt,
            base_url="http://localhost:8080/v1",
            api_key="none",
            model="qwen3-8b-q4",
            timeout=60,
        )

    def _query_localai(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """LocalAI — full stack, embedding+chat+vision, unlimited."""
        return self._query_openai_compatible(
            prompt, system_prompt,
            base_url="http://localhost:8082/v1",
            api_key="none",
            model="llama-3.3-70b",
            timeout=60,
        )

    # ------------------------------------------------------------------
    # OpenRouter: Qwen3-235B
    # ------------------------------------------------------------------
    def _query_openrouter_qwen235(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """OpenRouter Qwen3-235B — 50 req/day free, 235B params."""
        return self._query_openrouter(prompt, system_prompt, model="qwen/qwen3-235b-a22b:free")

    # ------------------------------------------------------------------
    # Chutes.ai — Decentralized Bittensor GPU network, DeepSeek R1 + Qwen3 free
    # ------------------------------------------------------------------
    def _query_chutes(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Chutes.ai — GPU decentralizzate Bittensor, DeepSeek-R1 + Qwen3-32B gratis."""
        api_key = settings.CHUTES_API_KEY
        if not api_key:
            return "", False, {"error": "no chutes key"}

        return self._query_openai_compatible(
            prompt, system_prompt,
            base_url="https://api.chutes.ai/v1",
            api_key=api_key,
            model="deepseek-ai/DeepSeek-R1",
            timeout=60,
        )

    # ------------------------------------------------------------------
    # Groq: Llama 4 Scout — 1K req/day separate quota
    # ------------------------------------------------------------------
    def _query_groq_llama4(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Groq Llama 4 Scout — 1K req/day separate quota, 10M context."""
        return self._query_groq_model(prompt, system_prompt, "meta-llama/llama-4-scout-17b-16e-instruct")

    # ------------------------------------------------------------------
    # Gemini Pro — gemini-2.5-pro for reasoning premium
    # ------------------------------------------------------------------
    def _query_gemini_pro(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """Gemini 2.5 Pro via OpenAI-compatible endpoint."""
        import os
        api_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return "", False, {"error": "GOOGLE_API_KEY non configurata"}
        return self._query_openai_compatible(
            prompt=prompt,
            system_prompt=system_prompt,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
            model="gemini-2.5-pro-preview-06-05",
            timeout=120,
        )
        if not api_key:
            return "", False, {}

        genai.configure(api_key=api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "user", "parts": [system_prompt]})
        messages.append({"role": "user", "parts": [prompt]})

        try:
            start = time.time()
            model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")
            response = model.generate_content(
                messages,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            latency = time.time() - start

            text = response.text or ""
            return text, bool(text.strip()), {
                "model": "gemini-2.5-pro",
                "provider": "google",
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            return "", False, {"error": str(e)}

    # ------------------------------------------------------------------
    # IO.NET Intelligence — 500K token/giorno gratis
    # ------------------------------------------------------------------
    def _query_ionet(self, prompt: str, system_prompt: str) -> Tuple[str, bool, Dict]:
        """IO.NET Intelligence — Llama-3.3-70B, 128K context, 500K token/giorno gratis."""
        import os
        api_key = os.environ.get("IONET_API_KEY", "")
        if not api_key:
            return "", False, {"error": "IONET_API_KEY non configurata"}
        
        return self._query_openai_compatible(
            prompt=prompt,
            system_prompt=system_prompt,
            base_url="https://api.intelligence.io.solutions/api/v1/",
            api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct",
            timeout=60,
        )
