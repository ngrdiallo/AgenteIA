"""
KeyRotator: rotazione di più API key per lo stesso provider.
Evita il rate limit usando chiavi diverse in sequenza.

V3 — asyncio.Lock con lazy initialization corretta.
asyncio.Lock() NON può essere creato a class level (import time).
Deve essere creato dentro l'event loop.
"""
import asyncio
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AllKeysExhaustedError(Exception):
    """Tutte le chiavi per un provider sono in cooldown."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"[{provider}] All keys exhausted or in cooldown")


class KeyState:
    """Stato di una singola chiave API."""
    
    def __init__(self, key: str, slot_id: int):
        self.key = key
        self.slot_id = slot_id
        self.cooldown_until = 0.0
        self.requests_this_minute = 0
        self.last_minute_reset = time.time()
        self.consecutive_429 = 0

    def is_available(self) -> bool:
        """True se la chiave non è in cooldown."""
        if time.time() < self.cooldown_until:
            return False
        if time.time() - self.last_minute_reset > 60:
            self.requests_this_minute = 0
            self.last_minute_reset = time.time()
        return True

    def use(self):
        """Incrementa contatore richieste."""
        self.requests_this_minute += 1

    def record_429(self):
        """Backoff esponenziale: 5m → 15m → 1h → 4h → 24h."""
        self.consecutive_429 += 1
        backoff = [300, 900, 3600, 14400, 86400]
        idx = min(self.consecutive_429 - 1, len(backoff) - 1)
        self.cooldown_until = time.time() + backoff[idx]
        logger.warning(
            f"Key slot {self.slot_id}: 429 #{self.consecutive_429}, "
            f"cooldown {backoff[idx]}s"
        )

    def record_success(self):
        """Reset consecutivi su successo."""
        self.consecutive_429 = 0


class KeyRotator:
    """
    Singleton per rotazione chiavi API.
    
    IMPORTANTE: usa asyncio.Lock con lazy initialization.
    asyncio.Lock() deve essere creato dentro l'event loop,
    NON a class level o import time.
    """
    _instance: Optional["KeyRotator"] = None

    def __new__(cls) -> "KeyRotator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._pools: dict[str, list[KeyState]] = {}
        self._pool_lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization del lock — sicuro nell'event loop."""
        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        return self._pool_lock

    async def register_keys(self, provider: str, keys: list[str]) -> None:
        """Registra chiavi per un provider."""
        valid = [k for k in keys if k and len(k.strip()) > 10]
        if not valid:
            logger.debug(f"KeyRotator: no valid keys for {provider}")
            return
        async with self._get_lock():
            self._pools[provider] = [
                KeyState(k, i) for i, k in enumerate(valid)
            ]
        logger.info(f"KeyRotator: {len(valid)} key(s) registered for {provider}")

    async def get_key(self, provider: str) -> tuple[str, int]:
        """Ottieni la chiave meno usata disponibile."""
        async with self._get_lock():
            pool = self._pools.get(provider, [])
            available = [ks for ks in pool if ks.is_available()]
            if not available:
                raise AllKeysExhaustedError(provider)
            best = min(available, key=lambda ks: ks.requests_this_minute)
            best.use()
            return best.key, best.slot_id

    async def report_429(self, provider: str, slot_id: int) -> None:
        """Registra un 429 per una chiave specifica."""
        async with self._get_lock():
            for ks in self._pools.get(provider, []):
                if ks.slot_id == slot_id:
                    ks.record_429()
                    return

    async def report_success(self, provider: str, slot_id: int) -> None:
        """Registra un successo — reset backoff."""
        async with self._get_lock():
            for ks in self._pools.get(provider, []):
                if ks.slot_id == slot_id:
                    ks.record_success()
                    return

    async def status(self) -> dict:
        """Stato real-time di tutte le chiavi."""
        async with self._get_lock():
            return {
                provider: [
                    {
                        "slot": ks.slot_id,
                        "available": ks.is_available(),
                        "cooldown_remaining_s": max(0, ks.cooldown_until - time.time()),
                        "rpm": ks.requests_this_minute,
                        "consecutive_429": ks.consecutive_429,
                    }
                    for ks in pool
                ]
                for provider, pool in self._pools.items()
            }


async def initialize_rotator_from_settings(settings) -> "KeyRotator":
    """Inizializza il KeyRotator con tutte le chiavi."""
    rotator = KeyRotator()

    xai_keys = [
        getattr(settings, "XAI_API_KEY", ""),
        getattr(settings, "XAI_API_KEY_2", ""),
    ]
    await rotator.register_keys("xai", [k for k in xai_keys if k])

    single_key_providers = {
        "groq":       getattr(settings, "GROQ_API_KEY", ""),
        "mistral":    getattr(settings, "MISTRAL_API_KEY", ""),
        "cerebras":   getattr(settings, "CEREBRAS_API_KEY", ""),
        "nvidia":     getattr(settings, "NVIDIA_API_KEY", ""),
        "fireworks":  getattr(settings, "FIREWORKS_API_KEY", ""),
        "gemini":     getattr(settings, "GOOGLE_API_KEY", ""),
        "openai":     getattr(settings, "OPENAI_API_KEY", ""),
        "sambanova":  getattr(settings, "SAMBANOVA_API_KEY", ""),
        "openrouter": getattr(settings, "OPENROUTER_API_KEY", ""),
        "deepseek":   getattr(settings, "DEEPSEEK_API_KEY", ""),
        "github":     getattr(settings, "GITHUB_TOKEN", ""),
    }

    for provider, key in single_key_providers.items():
        if key:
            await rotator.register_keys(provider, [key])

    return rotator


if __name__ == "__main__":
    async def smoke_test():
        rotator = KeyRotator()
        await rotator.register_keys("xai", ["fake_key_1_xxxxx", "fake_key_2_yyyyy"])

        k1, s1 = await rotator.get_key("xai")
        k2, s2 = await rotator.get_key("xai")
        assert s1 != s2, f"Round-robin fallito"
        print(f"PASS: round-robin OK (slot {s1} → slot {s2})")

        await rotator.report_429("xai", 0)
        k3, s3 = await rotator.get_key("xai")
        assert s3 == 1, f"Dopo 429 su slot 0, deve usare slot 1"
        print(f"PASS: fallback dopo 429 → slot {s3}")

        await rotator.report_429("xai", 1)
        try:
            await rotator.get_key("xai")
            print("FAIL: doveva sollevare AllKeysExhaustedError")
        except AllKeysExhaustedError:
            print("PASS: AllKeysExhaustedError sollevato")

        print("=== TUTTI I TEST PASSATI ===")

    asyncio.run(smoke_test())
