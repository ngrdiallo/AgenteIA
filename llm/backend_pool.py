"""
BackendPool: singleton che gestisce lo stato di tutti i provider LLM.

V2 - Correzioni:
- ZERO threading.Lock — solo asyncio.Lock()
- HTTP 400 context_length_exceeded NON conta come consecutive_failure
- HTTP 429 attiva Circuit Breaker sulla chiave, non sul provider intero
- Model Federation per fallback intelligente
- Weighted Router per scoring provider
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Stati del Circuit Breaker."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DEAD = "dead"
    RATE_LIMITED = "rate_limited"  # 429: provider ok, quota temporaneamente esaurita


@dataclass
class BackendState:
    """Stato di un singolo backend LLM."""
    name: str
    state: CircuitState = CircuitState.HEALTHY
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    base_cooldown: int = 25
    avg_latency_ms: float = 0.0
    success_count: int = 0
    total_calls: int = 0
    rate_limited_until: float = 0.0  # timestamp Unix, 0 = non in cooldown rate limit
    
    # Per weighted routing
    success_rate: float = 1.0
    weight: float = 1.0
    
    def is_ready(self) -> bool:
        """True se il provider è disponibile o se il cooldown è scaduto."""
        import time
        # Check rate limit first
        if self.state == CircuitState.RATE_LIMITED:
            if time.time() >= self.rate_limited_until:
                self.state = CircuitState.HEALTHY
                self.rate_limited_until = 0.0
                logger.info(f"✅ {self.name}: rate limit cooldown expired, back to healthy")
                return True
            return False  # still in rate limit cooldown
        if self.state == CircuitState.HEALTHY:
            return True
        if self.state == CircuitState.DEAD:
            elapsed = time.time() - self.last_failure_time
            wait = min(25 * (2 ** min(self.consecutive_failures, 5)), 120)
            if elapsed > wait:
                self.state = CircuitState.HEALTHY
                return True
            return False
        # DEGRADED - ok to try
        return True
    
    def record_success(self, latency_ms: float):
        """Registra un successo con latenza."""
        self.state = CircuitState.HEALTHY
        self.consecutive_failures = 0
        self.base_cooldown = 60
        self.success_count += 1
        self.total_calls += 1
        self.success_rate = self.success_count / max(self.total_calls, 1)
        
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * 0.8) + (latency_ms * 0.2)
    
    def record_failure(self, error_type: str = "generic", is_context_error: bool = False, retry_after: float = None):
        """
        Registra un fallimento.
        
        V2 CORREZIONE: HTTP 400 context_length NON incrementa consecutive_failures
        perché è un errore della richiesta, non del provider.
        
        retry_after: se specificato (da rate limit 429), imposta il cooldown esatto.
        """
        # Context length exceeded - errore del client, NON del provider
        if is_context_error:
            logger.debug(f"⏭️ {self.name}: context error (not counted as failure)")
            return
        
        self.consecutive_failures += 1
        # Se retry_after è specificato, imposta il cooldown esatto
        if retry_after:
            # Imposta last_failure_time indietro così che il cooldown termini al retry_after
            self.last_failure_time = time.time() - (self.base_cooldown - retry_after)
            logger.info(f"⏳ {self.name}: rate limit, cooldown fino a {retry_after:.0f}s")
        else:
            self.last_failure_time = time.time()
        self.total_calls += 1
        self.success_rate = self.success_count / max(self.total_calls, 1)
        
        # Calcola nuovo stato del circuit breaker
        # Soglie diverse per provider veloci vs lenti
        DEAD_THRESHOLDS = {
            "groq": 3,       # Fallisce per rate limit chiaro
            "cerebras": 3,  # Molto veloce
            "mistral": 8,   # Lento, vale la pena riprovare
            "nvidia": 8,    # Variabile
            "openrouter": 8, # Variabile
            "deepseek": 8,  # Lento ma stabile
            "gemini": 6,    # Affidabile
            "sambanova": 6,
            "fireworks": 6,
            "cloudflare": 6,
            "aimlapi": 6,
            "github": 6,
            "ovh": 6,
            "scaleway": 6,
            # Variants / aliases (più tolleranti per run lunghi)
            "openrouter_r1": 12,
            "openrouter_llama": 12,
            "openrouter_llama4": 12,
            "openrouter_qwen235": 12,
            "openrouter_gemma": 12,
            "groq_qwen3": 10,
            "groq_kimi": 10,
            "groq_gptoss": 10,
            "groq_llama4": 10,
            "cerebras_70b": 10,
            "cerebras_llama4": 10,
            "cerebras_qwen32": 10,
            "chutes": 12,
            "ionet": 10,
        }
        dead_threshold = DEAD_THRESHOLDS.get(self.name, 8)
        
        if self.state == CircuitState.HEALTHY and self.consecutive_failures >= 3:
            self.state = CircuitState.DEGRADED
            logger.warning(f"🟡 {self.name}: degraded after {self.consecutive_failures} failures")
        elif self.state == CircuitState.DEGRADED and self.consecutive_failures >= dead_threshold:
            self.state = CircuitState.DEAD
            logger.warning(f"🔴 {self.name}: DEAD after {self.consecutive_failures} failures (threshold={dead_threshold})")
    
    def record_rate_limit(self, cooldown_seconds: int = 60) -> None:
        """
        Registra un 429 rate limit.
        NON incrementa consecutive_failures — il provider è sano, solo esaurito.
        Mette il provider in cooldown per cooldown_seconds, poi auto-recovery.
        """
        import time
        self.state = CircuitState.RATE_LIMITED
        self.consecutive_failures = 0
        self.rate_limited_until = time.time() + cooldown_seconds
        self.total_calls += 1
        self.last_failure_time = time.time()
        logger.info(f"⏳ {self.name}: rate limited, cooldown {cooldown_seconds}s")
    
    def is_available(self) -> bool:
        """Verifica se il provider è disponibile (non dead, non in cooldown rate limit)."""
        import time
        if self.state == CircuitState.RATE_LIMITED:
            if time.time() >= self.rate_limited_until:
                # Cooldown scaduto → torna healthy automaticamente
                self.state = CircuitState.HEALTHY
                self.rate_limited_until = 0.0
                logger.info(f"✅ {self.name}: rate limit cooldown expired, back to healthy")
                return True
            return False  # ancora in cooldown
        
        if self.state == CircuitState.DEAD:
            # HALF-OPEN: dopo 300s dall'ultimo fallimento, riprova
            _RECOVERY_TIMEOUT = 300  # 5 minuti
            if (self.last_failure_time > 0 and
                    time.time() - self.last_failure_time >= _RECOVERY_TIMEOUT):
                self.state = CircuitState.DEGRADED
                self.consecutive_failures = max(0, 3)  # Reset a livello degraded
                logger.info(f"🔄 {self.name}: HALF-OPEN dopo {_RECOVERY_TIMEOUT}s")
            return self.state != CircuitState.DEAD
        
        return True  # HEALTHY o DEGRADED
    
    def get_weight(self) -> float:
        """
        Calcola peso per weighted routing.
        DEGRADED: weight * 0.3
        DEAD: excluded (weight = 0)
        """
        if self.state == CircuitState.DEAD:
            return 0.0
        if self.state == CircuitState.DEGRADED:
            return self.weight * 0.3
        # HEALTHY: full weight based on success rate and latency
        latency_score = 1.0 / (1.0 + self.avg_latency_ms / 1000)
        return self.success_rate * 0.5 + latency_score * 0.5


class BackendPool:
    """
    Singleton che gestisce lo stato di tutti i backend LLM.
    V2: usa asyncio.Lock per compatibilità con FastAPI.
    """
    _instance = None
    _init_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._state_lock = asyncio.Lock()
        
        # Context window (token) for each provider - synced with orchestrator _BACKEND_INPUT_LIMITS
        self.context_limits = {
            # Tier 1: High capacity (>=100k)
            "mistral":       120_000,
            "gemini":        900_000,
            "gemini_pro":  1_000_000,
            "gemini_lite":   300_000,
            "cerebras_qwen235": 200_000,
            "cerebras_llama4": 10_000_000,
            "cerebras_70b":   128_000,
            "modelscope_qwen": 128_000,

            # Tier 2: Medium capacity (60k-100k)
            "nvidia":        32_000,   # Note: actual limit varies by model
            "sambanova":     60_000,
            "openrouter":     60_000,
            "deepseek":       60_000,
            "chutes":         64_000,
            "fireworks":     131_072,

            # Tier 3: Lower capacity (<60k)
            "groq":          10_000,
            "groq_8b":       10_000,
            "groq_qwen3":    32_000,
            "groq_kimi":     32_000,
            "groq_gptoss":   32_000,
            "groq_llama4": 10_000_000,
            "cerebras":       8_000,
            "cerebras_qwen32": 32_000,
            "cloudflare":     32_000,
            "github":        128_000,
            "aimlapi":       128_000,
            "ovh":            32_000,
            "scaleway":       32_000,
            "ollama":          8_000,
            "huggingface":    8_000,
            "localai":        8_192,
            "llamacpp":       8_192,
            "vllm":          32_768,

            # Cloud variants (via Ollama endpoints)
            "glm5_cloud":    128_000,
            "kimi_cloud":    128_000,
            "minimax_cloud": 128_000,
            "openrouter_r1": 128_000,
            "openrouter_llama": 128_000,
            "openrouter_llama4": 10_000_000,
            "openrouter_qwen235": 200_000,
            "openrouter_gemma": 128_000,

            # New providers S24
            "nebius":        128_000,
            "ionet":         128_000,
            "cortecs":       128_000,
        }
        
        # Model Federation: family -> list of provider names
        self.model_federation = {
            "llama-3.1": ["nvidia", "fireworks", "groq", "cerebras", "mistral"],
            "llama-3.3": ["nvidia", "fireworks"],
            "mixtral": ["groq", "mistral"],
            "mistral-large": ["mistral"],
            "gemini-1.5": ["gemini"],
            "gpt-4o": ["openai"],
        }
        
        self._states = {
            name: BackendState(name=name)
            for name in self.context_limits
        }
        
        logger.info(
            f"🏊 BackendPool inizializzato con {len(self._states)} provider"
        )
    
    async def report_success_async(self, name: str, latency_ms: float):
        """Versione async-safe di report_success."""
        async with self._state_lock:
            if name in self._states:
                self._states[name].record_success(latency_ms)
    
    async def report_failure_async(self, name: str, error_type: str = "generic", is_context_error: bool = False, retry_after: float = None):
        """Versione async-safe di report_failure. V2: supporta is_context_error e retry_after."""
        async with self._state_lock:
            if name in self._states:
                self._states[name].record_failure(error_type, is_context_error, retry_after)
    
    def _reset_for_test(self):
        """Resetta tutti gli stati — per test unitari senza distruggere il singleton."""
        # Sincrono per test
        for name in list(self._states.keys()):
            self._states[name] = BackendState(name=name)

    def get_chain(self, routing_hint: str, token_estimate: int) -> list:
        """
        Ritorna la lista ordinata di provider pronti per il routing hint dato.
        Esclude automaticamente provider in cooldown o con contesto insufficiente.
        Se TUTTI i provider sono in cooldown, li include comunque come last resort
        (meglio un tentativo lento che nessuna risposta).
        """
        priority_map = {
            # Speed: Cerebras (illimitato) + Groq (veloce ma quota limitata)
            "speed": [
                "cerebras",             # llama3.1-8b, 1800 tok/s
                "cerebras_qwen32",      # Qwen3-32B, ~1400 tok/s
                "minimax_cloud",        # fast + agentic, Ollama cloud FREE
                "groq_8b",             # LPU sub-2s, 1M/day
                "groq",                # LPU sub-3s, 1M/day
                "gemini_lite",         # 1000 RPD
                "gemini_pro",          # 100 RPD, reasoning premium
                "cloudflare",          # edge, 10K/day
                "mistral",             # backup
                "localai",             # locale full stack
                "llamacpp",            # locale CPU
                "ollama",              # locale
                "ollama_free_api",      # distributed, no key
            ],
            # Reasoning: modelli qualitativi
            "reasoning": [
                "groq_qwen3",          # Qwen3-32B /think LPU (stabile)
                "gemini",              # 2.5 Flash
                "mistral",             # 1B token/mese
                "gemini_pro",          # 100 RPD, reasoning premium
                "nvidia",              # Llama 70B
                "groq_kimi",           # Kimi-K2 su Groq
                "groq_gptoss",         # GPT-OSS-120B
                "sambanova",           # 70B
                "cerebras",            # ultra-rapido
                "groq",                # veloce ma quota limitata
                # Demoted: tendono a saturare quota o diventare dead in run lunghi
                "openrouter_r1",       # DeepSeek R1 SOTA
                "chutes",              # DeepSeek-R1 decentralized
                # Experimental cloud variants
                "glm5_cloud",          # GLM-5 744B
                "kimi_cloud",          # Kimi K2.5
                "modelscope_qwen",     # Qwen3-480B
                "deepseek",            # DeepSeek V3
            ],
            # Large context: backbone Mistral + cloud reasoning + locale illimitato
            "large_context": [
                # TIER 1: backbone alto volume/stabilità
                "gemini_lite",         # 1000 RPD (PRIMA!)
                "gemini",              # 20 RPD (DOPO lite!)
                "mistral",             # 33M token/mese BACKBONE
                "groq_qwen3",          # Qwen3-32B Groq LPU 1K/day
                "groq_kimi",           # Kimi-K2 Groq 1K/day
                "groq_gptoss",         # GPT-OSS-120B 1K/day Groq
                "gemini_pro",          # 100 RPD, reasoning premium
                "ionet",               # IO.NET 500K token/day gratis ILLIMITATO
                "nvidia",              # 40 RPM
                "sambanova",           # Llama 3.3 70B
                "fireworks",           # Fireworks AI

                # TIER 2: fallback distribuito (più variabile)
                "groq_llama4",         # Llama4 Scout Groq 1K/day
                "openrouter_r1",       # DeepSeek R1 :free
                "chutes",              # Bittensor decentralized
                "openrouter_llama4",   # Llama4 10M ctx :free
                "openrouter_qwen235",  # Qwen3-235B :free
                "openrouter_qwen",     # Qwen 2.5 72B :free
                "openrouter_gemma",    # Gemma 3 27B
                "glm5_cloud",          # GLM-5 744B Ollama cloud
                "kimi_cloud",          # Kimi K2.5 Ollama cloud
                "minimax_cloud",       # MiniMax M2.5 Ollama cloud
                "cloudflare",          # 10K req/day
                "modelscope_qwen",     # Qwen3-480B 2K/day

                # TIER 3: capacità distribuita
                "cerebras_70b",        # Llama3.3-70B Cerebras 1M/day
                "cerebras_llama4",     # Llama4 Scout Cerebras 1M/day

                # TIER 4: safety net cloud
                "groq_8b",             # Llama3.1-8B Groq 1M/day
                "groq",                # Llama3.3-70B Groq 1M/day
                "openrouter_llama",     # Llama3.3-70B OR :free

                # TIER 5: locale infinito
                "vllm",               # locale GPU illimitato
                "localai",            # locale full stack
                "ollama_free_api",    # distributed, no key
                "llamacpp",           # locale CPU illimitato
                "huggingface",        # fallback
                "ollama",             # locale infinito
            ],
        }
        priority = priority_map.get(
            routing_hint,
            ["groq", "gemini", "cerebras", "mistral"]
        )

        # Sync access for read-only operation (states are thread-safe for reads)
        # Solo provider pronti con contesto sufficiente
        candidates = [
            name for name in priority
            if name in self._states
            and self._states[name].is_ready()
            and self.context_limits.get(name, 0) >= token_estimate
        ]
        # Aggiungi provider non in priority list come fallback
        for name, state in self._states.items():
            if (
                name not in candidates
                and state.is_ready()
                and self.context_limits.get(name, 0) >= token_estimate
            ):
                candidates.append(name)

        # LAST RESORT: se tutti in cooldown, includi comunque quelli
        # con contesto sufficiente (ordinati per priority)
        if not candidates:
            logger.warning(
                "⚠️ Tutti i provider in cooldown — "
                "forzando last-resort chain"
            )
            candidates = [
                name for name in priority
                if name in self._states
                and self.context_limits.get(name, 0) >= token_estimate
            ]
            for name in self._states:
                if (
                    name not in candidates
                    and self.context_limits.get(name, 0) >= token_estimate
                ):
                    candidates.append(name)

        return candidates

    def report_success(self, name: str, latency_ms: float):
        """Registra successo per un provider (sync version for tests)."""
        if name in self._states:
            self._states[name].record_success(latency_ms)

    def report_failure(self, name: str, error_type: str = "generic", is_context_error: bool = False, retry_after: float = None):
        """Registra fallimento per un provider (sync version for tests)."""
        if name in self._states:
            self._states[name].record_failure(error_type, is_context_error, retry_after)

    def record_rate_limit(self, name: str, cooldown_seconds: int = 60, error_msg: str = ""):
        """
        Registra un 429 rate limit per un provider.
        NON incrementa consecutive_failures — il provider è sano, solo esaurito.
        Mette il provider in cooldown per cooldown_seconds, poi auto-recovery.
        """
        if name in self._states:
            err_lower = (error_msg or "").lower()
            daily_kw = (
                "daily quota",
                "per day",
                "daily limit",
                "limit_rpd",
                "daily_limit",
                "daily cap",
                "requests per day",
                "rate_limit_exceeded",
                "per-day",
                "rpd",
            )
            effective_cooldown = 86400 if any(kw in err_lower for kw in daily_kw) else cooldown_seconds
            self._states[name].record_rate_limit(effective_cooldown)

    def status(self) -> dict:
        """Stato real-time di tutti i provider — per debug e UI."""
        return {
            name: {
                "ready": state.is_ready(),
                "state": state.state.value,
                "consecutive_failures": state.consecutive_failures,
                "avg_latency_ms": round(state.avg_latency_ms),
                "success_rate": round(state.success_rate, 2),
                "total_calls": state.total_calls,
                "weight": round(state.get_weight(), 2),
            }
            for name, state in self._states.items()
        }

    @classmethod
    def reset(cls):
        """Reset singleton — solo per i test."""
        cls._instance = None
