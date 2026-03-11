"""
Verifica REALE di ogni provider — fa una chiamata HTTP e misura la latenza.
Esegui: python scripts/verify_providers.py
"""
import asyncio
import time
import httpx
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import settings

TEST_PROMPT = "Rispondi SOLO con: OK"
SYSTEM_PROMPT = "Sei un assistente. Rispondi solo con OK."

async def test_provider(name: str, test_fn) -> dict:
    start = time.time()
    try:
        result = await asyncio.wait_for(test_fn(), timeout=15.0)
        latency = (time.time() - start) * 1000
        return {"name": name, "ok": True, "latency_ms": latency, "response": str(result)[:50]}
    except asyncio.TimeoutError:
        return {"name": name, "ok": False, "error": "TIMEOUT >15s"}
    except Exception as e:
        return {"name": name, "ok": False, "error": str(e)[:100]}

async def test_groq():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_cerebras():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.CEREBRAS_API_KEY}"},
            json={"model": "llama3.1-8b",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_gemini():
    key = settings.GOOGLE_API_KEY
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
            json={"contents": [{"parts": [{"text": TEST_PROMPT}]}]}
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

async def test_openai():
    key = settings.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "gpt-4o-mini",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_xai():
    key = settings.XAI_API_KEY
    if not key:
        raise ValueError("XAI_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "grok-3-mini",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_xai_2():
    key = settings.XAI_API_KEY_2
    if not key:
        raise ValueError("XAI_API_KEY_2 mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "grok-3-mini",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_nvidia():
    key = settings.NVIDIA_API_KEY
    if not key:
        raise ValueError("NVIDIA_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "meta/llama-3.1-8b-instruct",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_hyperbolic():
    key = settings.HYPERBOLIC_API_KEY
    if not key:
        raise ValueError("HYPERBOLIC_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.hyperbolic.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "meta-llama/Llama-3.3-70B-Instruct",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_fireworks():
    key = settings.FIREWORKS_API_KEY
    if not key:
        raise ValueError("FIREWORKS_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "accounts/fireworks/models/llama-v3p3-70b-instruct",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_chutes():
    key = settings.CHUTES_API_KEY
    if not key:
        raise ValueError("CHUTES_API_KEY mancante")
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://llm.chutes.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": "deepseek-ai/DeepSeek-V3-0324",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5, "stream": False}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_mistral():
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.MISTRAL_API_KEY}"},
            json={"model": "mistral-small-latest",
                  "messages": [{"role": "user", "content": TEST_PROMPT}],
                  "max_tokens": 5}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def test_tavily():
    key = settings.TAVILY_API_KEY
    if not key:
        raise ValueError("TAVILY_API_KEY mancante")
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            "https://api.tavily.com/search",
            json={"api_key": key, "query": "test", "max_results": 1}
        )
        r.raise_for_status()
        return f"{len(r.json().get('results', []))} results"

TESTS = [
    ("groq",        test_groq),
    ("cerebras",    test_cerebras),
    ("gemini",      test_gemini),
    ("openai",      test_openai),
    ("xai_key1",    test_xai),
    ("xai_key2",    test_xai_2),
    ("nvidia",      test_nvidia),
    ("hyperbolic",  test_hyperbolic),
    ("fireworks",   test_fireworks),
    ("chutes",      test_chutes),
    ("mistral",     test_mistral),
    ("tavily",      test_tavily),
]

async def main():
    print("=" * 60)
    print("VERIFICA PROVIDER — test con chiamata HTTP reale")
    print("=" * 60)
    
    results = []
    for name, fn in TESTS:
        print(f"  Testing {name}...", end=" ", flush=True)
        r = await test_provider(name, fn)
        results.append(r)
        if r["ok"]:
            print(f"OK ({r['latency_ms']:.0f}ms) → {r['response']}")
        else:
            print(f"FAIL: {r['error']}")
    
    print("\n" + "=" * 60)
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"FUNZIONANTI: {len(ok)}/{len(results)}")
    print(f"FALLITI: {[r['name'] for r in fail]}")
    print("=" * 60)
    
    # Salva risultati in un file per uso successivo
    import json
    with open("provider_status.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRisultati salvati in provider_status.json")

asyncio.run(main())
