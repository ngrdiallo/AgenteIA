"""
SUBNP Client: wrapper per SubNP - provider sperimentale.

V2: ZERO import dal core - completamente isolato.
Timeout hardcoded: 3 secondi.
Cache interna: 5 minuti.
Mai sollevare eccezioni verso il chiamante.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# CONSTANTS
# ============================================================

SUBNP_BASE_URL = "https://api.subnp.com/v1"
TIMEOUT_SECONDS = 3
CACHE_TTL_SECONDS = 300  # 5 minuti


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SubNPResponse:
    """Risposta da SubNP."""
    content: str
    model: str
    latency_ms: float
    cached: bool = False


# ============================================================
# CLIENT
# ============================================================

class SubNPClient:
    """
    Client per SubNP - provider sperimentale.
    
    V2: ISOLATO - zero import da core.
    """
    
    def __init__(self, api_key: str, base_url: str = SUBNP_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self._cache: dict[str, tuple[str, float]] = {}
        self._client: Optional[any] = None
    
    async def _get_client(self):
        """Lazy import di httpx."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
        return self._client
    
    def _make_cache_key(self, prompt: str, model: str) -> str:
        """Crea chiave cache."""
        raw = f"{model}:{prompt[:100]}"
        return hash(raw)
    
    async def complete(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
    ) -> SubNPResponse:
        """
        Completa prompt tramite SubNP.
        
        NEVER raises - returns empty on failure.
        """
        cache_key = self._make_cache_key(prompt, model)
        now = time.time()
        
        # Check cache
        if cache_key in self._cache:
            cached_content, cached_time = self._cache[cache_key]
            if now - cached_time < CACHE_TTL_SECONDS:
                return SubNPResponse(
                    content=cached_content,
                    model=model,
                    latency_ms=0,
                    cached=True
                )
        
        start_time = now
        
        try:
            import httpx
            
            client = await self._get_client()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                logger.warning(f"SubNP returned {response.status_code}")
                return SubNPResponse(
                    content="",
                    model=model,
                    latency_ms=latency_ms
                )
            
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Cache result
            self._cache[cache_key] = (content, now)
            
            return SubNPResponse(
                content=content,
                model=model,
                latency_ms=latency_ms,
                cached=False
            )
            
        except asyncio.TimeoutError:
            logger.warning("SubNP timeout")
            return SubNPResponse(
                content="",
                model=model,
                latency_ms=TIMEOUT_SECONDS * 1000
            )
        except Exception as e:
            logger.error(f"SubNP error: {e}")
            return SubNPResponse(
                content="",
                model=model,
                latency_ms=0
            )
    
    async def health_check(self) -> dict:
        """
        Health check - ritorna disponibilità.
        """
        try:
            result = await self.complete("ping", model="test")
            return {
                "available": bool(result.content),
                "latency_ms": result.latency_ms,
            }
        except Exception:
            return {"available": False, "latency_ms": 0}
    
    async def close(self):
        """Cleanup."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_client: Optional[SubNPClient] = None


def get_subnp_client(api_key: str) -> SubNPClient:
    """Ritorna client SubNP singleton."""
    global _client
    if _client is None:
        _client = SubNPClient(api_key)
    return _client


# SMOKE TEST
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("=== SubNP Client Test ===")
        
        # Test with dummy key (will fail, but tests structure)
        client = SubNPClient("dummy-key")
        
        # Test health check
        health = await client.health_check()
        print(f"Health: {health}")
        
        await client.close()
    
    asyncio.run(test())
