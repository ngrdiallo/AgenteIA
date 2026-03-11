"""
Response Cache: caching deterministico per risposte LLM.

TTL:
- meta: 3600s (1 ora)
- focused: 900s (15 minuti)
- comprehensive: NO CACHE (risposte troppo variabili)

Key: SHA256(intent + sorted_words(normalize(query)) + collection_id)
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry nella cache."""
    response: str
    timestamp: float
    ttl: int  # seconds


class ResponseCache:
    """
    LRU cache con TTL per risposte LLM.
    V2: usa asyncio.Lock per compatibilità con FastAPI.
    """
    
    def __init__(self, max_entries: int = 500, flush_interval: int = 300):
        self.max_entries = max_entries
        self.flush_interval = flush_interval
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._last_flush = time.time()
        
        # TTL per intent
        self._ttl_by_intent = {
            "meta": 3600,
            "greeting": 3600,
            "focused": 900,
            "comprehensive": 0,  # NO CACHE
        }
    
    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalizza query per cache key consistente."""
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', ' ', q)
        q = re.sub(r'\s+', ' ', q)
        words = sorted(q.split())
        return ' '.join(words)
    
    def _make_key(self, intent: str, query: str, collection_id: str) -> str:
        """
        Crea chiave cache deterministica.
        SHA256(intent + normalized_query + collection_id)
        """
        normalized = self._normalize_query(query)
        raw = f"{intent}:{normalized}:{collection_id}"
        return hashlib.sha256(raw.encode()).hexdigest()
    
    async def get(self, intent: str, query: str, collection_id: str) -> Optional[str]:
        """
        Ritorna cached response se presente e non scaduta.
        """
        # Comprehensive = NO CACHE
        if intent == "comprehensive":
            return None
        
        key = self._make_key(intent, query, collection_id)
        ttl = self._ttl_by_intent.get(intent, 900)
        
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            age = time.time() - entry.timestamp
            
            if age > ttl:
                del self._cache[key]
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            logger.debug(f"📦 Cache HIT: {intent} (age: {age:.1f}s)")
            return entry.response
    
    async def set(self, intent: str, query: str, collection_id: str, response: str):
        """
        Salva response in cache.
        """
        # Comprehensive = NO CACHE
        if intent == "comprehensive":
            return
        
        key = self._make_key(intent, query, collection_id)
        
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                response=response,
                timestamp=time.time(),
                ttl=self._ttl_by_intent.get(intent, 900)
            )
            self._cache.move_to_end(key)
            
            logger.debug(f"📦 Cache SET: {intent}")
    
    async def invalidate(self, collection_id: str):
        """Invalidates all cache entries for a collection."""
        async with self._lock:
            to_remove = [
                k for k, v in self._cache.items()
                if collection_id in k
            ]
            for k in to_remove:
                del self._cache[k]
            logger.info(f"🗑️ Invalidated {len(to_remove)} cache entries")
    
    async def flush_if_needed(self, filepath: str = "data/response_cache.json"):
        """
        Flush cache to JSON if interval elapsed.
        """
        now = time.time()
        if now - self._last_flush < self.flush_interval:
            return
        
        async with self._lock:
            data = {
                key: {
                    "response": entry.response,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                }
                for key, entry in self._cache.items()
            }
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(data, f)
                self._last_flush = now
                logger.info(f"💾 Flushed {len(data)} cache entries to {filepath}")
            except Exception as e:
                logger.error(f"Failed to flush cache: {e}")
    
    async def load_from_file(self, filepath: str = "data/response_cache.json"):
        """Load cache from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            now = time.time()
            async with self._lock:
                for key, entry_data in data.items():
                    age = now - entry_data["timestamp"]
                    if age < entry_data["ttl"]:
                        self._cache[key] = CacheEntry(
                            response=entry_data["response"],
                            timestamp=entry_data["timestamp"],
                            ttl=entry_data["ttl"]
                        )
            logger.info(f"📂 Loaded {len(self._cache)} cache entries from {filepath}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    async def stats(self) -> dict:
        """Ritorna statistiche cache."""
        async with self._lock:
            now = time.time()
            active = sum(
                1 for e in self._cache.values()
                if now - e.timestamp < e.ttl
            )
            return {
                "total_entries": len(self._cache),
                "active_entries": active,
                "max_entries": self.max_entries,
                "last_flush": self._last_flush,
            }


# Global cache instance
_response_cache: Optional[ResponseCache] = None


def get_response_cache() -> ResponseCache:
    """Ritorna l'istanza globale della cache."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


# SMOKE TEST
if __name__ == "__main__":
    import asyncio
    
    async def test():
        cache = ResponseCache(max_entries=10)
        
        # Test cache miss
        result = await cache.get("meta", "Ciao", "default")
        print(f"Miss: {result}")
        
        # Test cache set
        await cache.set("meta", "Ciao", "default", "Ciao! Come posso aiutarti?")
        
        # Test cache hit
        result = await cache.get("meta", "Ciao", "default")
        print(f"Hit: {result}")
        
        # Test stats
        stats = await cache.stats()
        print(f"Stats: {stats}")
        
        # Test comprehensive = NO CACHE
        await cache.set("comprehensive", "Analizza questo documento...", "default", "Analisi completa...")
        result = await cache.get("comprehensive", "Analizza questo documento...", "default")
        print(f"Comprehensive (should be None): {result}")
    
    asyncio.run(test())
