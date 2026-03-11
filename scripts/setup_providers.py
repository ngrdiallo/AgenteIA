#!/usr/bin/env python3
"""
Setup Providers CLI: gestisci provider LLM da linea di comando.

Usage:
  python scripts/setup_providers.py --dry-run
  python scripts/setup_providers.py --validate
  python scripts/setup_providers.py --add groq
  python scripts/setup_providers.py --list
"""
import argparse
import asyncio
import json
import sys
import os
from dataclasses import dataclass
from typing import Optional

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ProviderStatus:
    """Stato di un provider."""
    name: str
    tier: str  # "core" or "experimental"
    keys_valid: list[str]
    keys_failed: list[str]
    status: str  # "OK", "FAILED", "PARTIAL"


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

PROVIDERS = {
    "groq": {
        "tier": "core",
        "key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1/models",
        "test_model": "llama-3.1-8b-instant",
    },
    "cerebras": {
        "tier": "core",
        "key_env": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1/models",
        "test_model": "llama-3.1-8b",
    },
    "mistral": {
        "tier": "core",
        "key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1/models",
        "test_model": "mistral-small-latest",
    },
    "nvidia": {
        "tier": "core",
        "key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1/models",
        "test_model": "nvidia/llama-3.1-8b-instruct",
    },
    "fireworks": {
        "tier": "core",
        "key_env": "FIREWORKS_API_KEY",
        "base_url": "https://api.fireworks.ai/inference/v1/models",
        "test_model": "llama-3.1-70b-instruct",
    },
    "gemini": {
        "tier": "core",
        "key_env": "GOOGLE_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1/models",
        "test_model": "gemini-1.5-flash",
    },
    "ollama": {
        "tier": "core",
        "key_env": None,  # No API key needed for local
        "base_url": "http://localhost:11434/api/tags",
        "test_model": None,
    },
    "subnp": {
        "tier": "experimental",
        "key_env": "SUBNP_API_KEY",
        "base_url": "https://api.subnp.com/v1",
        "test_model": "default",
    },
    "xai": {
        "tier": "core",
        "key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1/models",
        "test_model": "grok-beta",
    },
}


# ============================================================
# FUNCTIONS
# ============================================================

def load_env() -> dict:
    """Load .env file."""
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip()
    
    return env


async def validate_provider(provider_name: str, config: dict, env: dict) -> ProviderStatus:
    """Valida un singolo provider."""
    key_env = config.get("key_env")
    tier = config.get("tier", "core")
    
    keys_valid = []
    keys_failed = []
    
    # Get keys (handle multi-key providers like xAI)
    if provider_name == "xai":
        keys = [env.get("XAI_API_KEY", ""), env.get("XAI_API_KEY_2", "")]
    else:
        keys = [env.get(key_env, "")] if key_env else []
    
    keys = [k for k in keys if k]  # Remove empty
    
    if not keys:
        return ProviderStatus(
            name=provider_name,
            tier=tier,
            keys_valid=[],
            keys_failed=[],
            status="NO_KEY"
        )
    
    # Test each key
    for key in keys:
        try:
            if config.get("base_url") and config.get("test_model"):
                import httpx
                client = httpx.AsyncClient(timeout=10)
                
                # Simple test - just check if we can list models
                headers = {"Authorization": f"Bearer {key}"}
                response = await client.get(config["base_url"], headers=headers)
                
                if response.status_code in (200, 201):
                    keys_valid.append(key[:10] + "...")
                else:
                    keys_failed.append(key[:10] + f"... ({response.status_code})")
                
                await client.aclose()
            else:
                # Local provider like Ollama - just mark as valid
                keys_valid.append(key[:10] + "..." if key else "local")
                
        except Exception as e:
            keys_failed.append(key[:10] + f"... ({str(e)[:20]})")
    
    if keys_valid and not keys_failed:
        status = "OK"
    elif keys_valid and keys_failed:
        status = "PARTIAL"
    else:
        status = "FAILED"
    
    return ProviderStatus(
        name=provider_name,
        tier=tier,
        keys_valid=keys_valid,
        keys_failed=keys_failed,
        status=status
    )


async def cmd_validate(args) -> int:
    """Valida tutti i provider."""
    env = load_env()
    
    print("\n=== Provider Validation ===\n")
    
    results = []
    for name, config in PROVIDERS.items():
        status = await validate_provider(name, config, env)
        results.append(status)
        
        tier_icon = "🟢" if status.tier == "core" else "🟡"
        status_icon = "✅" if status.status == "OK" else "⚠️" if status.status == "PARTIAL" else "❌"
        
        print(f"{tier_icon} {status.name:12} {status_icon} {status.status}")
        if status.keys_valid:
            print(f"   Valid: {', '.join(status.keys_valid)}")
        if status.keys_failed:
            print(f"   Failed: {', '.join(status.keys_failed)}")
    
    # Summary
    core_ok = sum(1 for r in results if r.tier == "core" and r.status == "OK")
    exp_ok = sum(1 for r in results if r.tier == "experimental" and r.status == "OK")
    
    print(f"\n=== Summary ===")
    print(f"Core providers OK: {core_ok}/{sum(1 for r in results if r.tier == 'core')}")
    print(f"Experimental OK: {exp_ok}/{sum(1 for r in results if r.tier == 'experimental')}")
    
    return 0


def cmd_list(args) -> int:
    """Lista tutti i provider."""
    print("\n=== Available Providers ===\n")
    print(f"{'Provider':<15} {'Tier':<12} {'Key Env':<20} {'Test Model'}")
    print("-" * 70)
    
    for name, config in PROVIDERS.items():
        tier = config.get("tier", "core")
        key_env = config.get("key_env", "N/A")
        model = config.get("test_model", "N/A")
        print(f"{name:<15} {tier:<12} {key_env:<20} {model}")
    
    return 0


def cmd_dry_run(args) -> int:
    """Mostra cosa farebbe --add."""
    print("\n=== Dry Run ===\n")
    print("Provider configuration would be:")
    print(json.dumps(PROVIDERS, indent=2))
    return 0


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Setup providers CLI")
    
    parser.add_argument("--validate", action="store_true", help="Validate all providers")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--add", metavar="PROVIDER", help="Add a provider")
    
    args = parser.parse_args()
    
    if args.validate:
        return asyncio.run(cmd_validate(args))
    elif args.list:
        return cmd_list(args)
    elif args.dry_run:
        return cmd_dry_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
