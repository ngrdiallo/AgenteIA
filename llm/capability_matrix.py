"""
Capability Matrix: modelli LLM con le loro capacità tecniche.
Usato per routing intelligente basato su requisiti (latency, tools, context).
"""
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapabilities:
    """Capacità tecniche di un modello."""
    model_id: str
    provider: str
    max_context_tokens: int
    supports_tool_calling: bool
    supports_streaming: bool
    supports_json: bool
    supports_vision: bool
    avg_latency_ms: float
    cost_per_1k_input: float
    cost_per_1k_output: float
    family: str  # e.g., "llama-3", "mixtral", "gemini"


CAPABILITY_MATRIX: dict[str, ModelCapabilities] = {
    # === GROQ (verificato OK) ===
    "groq/llama-3.1-70b-versatile": ModelCapabilities(
        model_id="llama-3.1-70b-versatile",
        provider="groq",
        max_context_tokens=131_072,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=800,
        cost_per_1k_input=0.0007,
        cost_per_1k_output=0.0008,
        family="llama-3.1"
    ),
    "groq/llama-3.1-8b-instant": ModelCapabilities(
        model_id="llama-3.1-8b-instant",
        provider="groq",
        max_context_tokens=131_072,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=350,
        cost_per_1k_input=0.00005,
        cost_per_1k_output=0.00008,
        family="llama-3.1"
    ),
    "groq/mixtral-8x7b-32768": ModelCapabilities(
        model_id="mixtral-8x7b-32768",
        provider="groq",
        max_context_tokens=32_768,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=400,
        cost_per_1k_input=0.00024,
        cost_per_1k_output=0.00024,
        family="mixtral"
    ),

    # === CEREBRAS (verificato OK) ===
    "cerebras/llama-3.1-8b": ModelCapabilities(
        model_id="llama-3.1-8b",
        provider="cerebras",
        max_context_tokens=8_192,
        supports_tool_calling=False,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=200,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0001,
        family="llama-3.1"
    ),
    "cerebras/llama-3.3-70b": ModelCapabilities(
        model_id="llama-3.3-70b",
        provider="cerebras",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=600,
        cost_per_1k_input=0.0006,
        cost_per_1k_output=0.0006,
        family="llama-3.3"
    ),

    # === MISTRAL (verificato OK) ===
    "mistral/mistral-large-latest": ModelCapabilities(
        model_id="mistral-large-latest",
        provider="mistral",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=1200,
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.006,
        family="mistral-large"
    ),
    "mistral/mistral-small-latest": ModelCapabilities(
        model_id="mistral-small-latest",
        provider="mistral",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=500,
        cost_per_1k_input=0.0002,
        cost_per_1k_output=0.0006,
        family="mistral-small"
    ),

    # === NVIDIA (verificato OK) ===
    "nvidia/llama-3.1-nemotron-70b-instruct": ModelCapabilities(
        model_id="llama-3.1-nemotron-70b-instruct",
        provider="nvidia",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=900,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0005,
        family="llama-3.1"
    ),
    "nvidia/llama-3.1-8b-instruct": ModelCapabilities(
        model_id="llama-3.1-8b-instruct",
        provider="nvidia",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=300,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0001,
        family="llama-3.1"
    ),

    # === FIREWORKS (verificato OK, ma lento) ===
    "fireworks/llama-3.1-70b-instruct": ModelCapabilities(
        model_id="llama-3.1-70b-instruct",
        provider="fireworks",
        max_context_tokens=131_072,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=2500,
        cost_per_1k_input=0.0007,
        cost_per_1k_output=0.0008,
        family="llama-3.1"
    ),
    "fireworks/llama-3.3-70b-instruct": ModelCapabilities(
        model_id="llama-3.3-70b-instruct",
        provider="fireworks",
        max_context_tokens=131_072,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=2800,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.0009,
        family="llama-3.3"
    ),

    # === GEMINI (verificato 400 Bad Request) ===
    "gemini/gemini-1.5-flash": ModelCapabilities(
        model_id="gemini-1.5-flash",
        provider="gemini",
        max_context_tokens=1_000_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=True,
        avg_latency_ms=600,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        family="gemini-1.5"
    ),
    "gemini/gemini-1.5-pro": ModelCapabilities(
        model_id="gemini-1.5-pro",
        provider="gemini",
        max_context_tokens=2_000_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=True,
        avg_latency_ms=1500,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        family="gemini-1.5"
    ),

    # === OPENAI (verificato 429 Rate Limit) ===
    "openai/gpt-4o-mini": ModelCapabilities(
        model_id="gpt-4o-mini",
        provider="openai",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=True,
        avg_latency_ms=400,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        family="gpt-4o"
    ),
    "openai/gpt-4o": ModelCapabilities(
        model_id="gpt-4o",
        provider="openai",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=True,
        avg_latency_ms=800,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        family="gpt-4o"
    ),

    # === OLLAMA (locale) ===
    "ollama/llama3.1": ModelCapabilities(
        model_id="llama3.1",
        provider="ollama",
        max_context_tokens=128_000,
        supports_tool_calling=False,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=100,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        family="llama-3.1"
    ),
    "ollama/phi3": ModelCapabilities(
        model_id="phi3",
        provider="ollama",
        max_context_tokens=4_096,
        supports_tool_calling=False,
        supports_streaming=True,
        supports_json=False,
        supports_vision=False,
        avg_latency_ms=50,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        family="phi-3"
    ),

    # === SAMBANOVA ===
    "sambanova/Llama-3.1-70B-Instruct": ModelCapabilities(
        model_id="Llama-3.1-70B-Instruct",
        provider="sambanova",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=1000,
        cost_per_1k_input=0.0003,
        cost_per_1k_output=0.0003,
        family="llama-3.1"
    ),

    # === DEEPSEEK ===
    "deepseek/deepseek-chat": ModelCapabilities(
        model_id="deepseek-chat",
        provider="deepseek",
        max_context_tokens=64_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=700,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        family="deepseek"
    ),

    # === OPENROUTER ===
    "openrouter/meta-llama/llama-3.1-70b-instruct": ModelCapabilities(
        model_id="meta-llama/llama-3.1-70b-instruct",
        provider="openrouter",
        max_context_tokens=128_000,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_json=True,
        supports_vision=False,
        avg_latency_ms=2000,
        cost_per_1k_input=0.0004,
        cost_per_1k_output=0.0004,
        family="llama-3.1"
    ),
}


class IntentRequirements:
    """Requisiti per tipo di intent."""
    
    META = {
        "max_latency_ms": 3000,
        "requires_tools": False,
        "requires_vision": False,
        "min_context": 0,
    }
    
    GREETING = {
        "max_latency_ms": 2000,
        "requires_tools": False,
        "requires_vision": False,
        "min_context": 0,
    }
    
    FOCUSED = {
        "max_latency_ms": 10000,
        "requires_tools": True,
        "requires_vision": False,
        "min_context": 4_000,
    }
    
    COMPREHENSIVE = {
        "max_latency_ms": 45_000,
        "requires_tools": True,
        "requires_vision": False,
        "min_context": 32_768,
    }


def get_models_for_intent(intent: str) -> list[ModelCapabilities]:
    """
    Filtra modelli in base ai requisiti dell'intent.
    
    Args:
        intent: one of "meta", "greeting", "focused", "comprehensive"
    
    Returns:
        Lista di ModelCapabilities ordinata per latenza (più veloce prima)
    """
    reqs = getattr(IntentRequirements, intent.upper(), IntentRequirements.FOCUSED)
    
    candidates = []
    for cap in CAPABILITY_MATRIX.values():
        if cap.avg_latency_ms > reqs["max_latency_ms"]:
            continue
        if reqs["requires_tools"] and not cap.supports_tool_calling:
            continue
        if reqs["requires_vision"] and not cap.supports_vision:
            continue
        if cap.max_context_tokens < reqs["min_context"]:
            continue
        candidates.append(cap)
    
    candidates.sort(key=lambda c: c.avg_latency_ms)
    return candidates


def get_model_by_id(model_id: str) -> Optional[ModelCapabilities]:
    """Ritorna capabilities per model_id specifico."""
    return CAPABILITY_MATRIX.get(model_id)


def get_models_by_family(family: str) -> list[ModelCapabilities]:
    """Ritorna tutti i modelli di una family (es. llama-3.1)."""
    return [c for c in CAPABILITY_MATRIX.values() if c.family == family]


def get_federation(family: str) -> dict[str, list[str]]:
    """
    Federation per model family: same model on different providers.
    Usato per fallback intelligente.
    """
    federation = {}
    for cap in CAPABILITY_MATRIX.values():
        if cap.family == family:
            key = f"{cap.provider}/{cap.model_id}"
            if family not in federation:
                federation[family] = []
            federation[family].append(key)
    return federation


# SMOKE TEST
if __name__ == "__main__":
    print("=== Capability Matrix ===")
    print(f"Total models: {len(CAPABILITY_MATRIX)}")
    
    meta_models = get_models_for_intent("meta")
    print(f"\nMeta intent models: {len(meta_models)}")
    for m in meta_models[:3]:
        print(f"  - {m.model_id} ({m.provider}): {m.avg_latency_ms}ms")
    
    focused_models = get_models_for_intent("focused")
    print(f"\nFocused intent models (with tools): {len(focused_models)}")
    for m in focused_models[:3]:
        print(f"  - {m.model_id} ({m.provider}): {m.avg_latency_ms}ms")
    
    comprehensive_models = get_models_for_intent("comprehensive")
    print(f"\nComprehensive intent models: {len(comprehensive_models)}")
    for m in comprehensive_models[:3]:
        print(f"  - {m.model_id} ({m.provider}): {m.max_context_tokens} tokens")
    
    # Test federation
    llama_family = get_federation("llama-3.1")
    print(f"\nLlama-3.1 Federation: {llama_family}")
