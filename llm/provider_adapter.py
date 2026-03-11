"""
Provider Adapter: astratto e implementazioni per ogni provider LLM.

V2 — Correzioni:
- ZERO eval() — sostituito con json.loads() ovunque
- KeyRotator iniettato nel costruttore
- Eccezioni typed sempre rilanciate (no swallow)
- HTTP 400 context → ProviderContextError (non incrementa Circuit Breaker)
- HTTP 429 → ProviderRateLimitError + report al KeyRotator
"""
import asyncio
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import httpx

from llm.key_rotator import KeyRotator, AllKeysExhaustedError

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderRateLimitError(ProviderError):
    def __init__(self, provider: str, key_slot: int, retry_after: int = 60):
        self.key_slot = key_slot
        self.retry_after = retry_after
        super().__init__(provider, f"Rate limit (429) - key slot {key_slot}")


class ProviderContextError(ProviderError):
    def __init__(self, provider: str, model: str, tokens_req: int = 0, tokens_max: int = 0):
        self.model = model
        self.tokens_req = tokens_req
        self.tokens_max = tokens_max
        super().__init__(provider, f"Context length exceeded: req={tokens_req} > max={tokens_max}")


class ProviderAuthError(ProviderError):
    def __init__(self, provider: str, key_slot: int):
        self.key_slot = key_slot
        super().__init__(provider, f"Auth failed (401/403) - key slot {key_slot}")


class ProviderTimeoutError(ProviderError):
    def __init__(self, provider: str, timeout_s: float):
        self.timeout_s = timeout_s
        super().__init__(provider, f"Timeout after {timeout_s}s")


@dataclass
class ToolCallResult:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    raw_response: Optional[dict] = None


class ProviderAdapter(ABC):
    def __init__(self, provider_name: str, key_rotator: KeyRotator, timeout: float = 30.0):
        self.provider_name = provider_name
        self.key_rotator = key_rotator
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    @abstractmethod
    def endpoint_url(self) -> str:
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        pass

    @abstractmethod
    def _build_headers(self, api_key: str) -> dict:
        pass

    @abstractmethod
    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        pass

    @abstractmethod
    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        pass

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), follow_redirects=True)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        tools: Optional[list[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        model = model or self.default_model
        start_time = time.time()

        api_key, slot_id = await self.key_rotator.get_key(self.provider_name)

        headers = self._build_headers(api_key)
        payload = self._build_payload(model, messages, temperature=temperature, max_tokens=max_tokens, tools=tools, **kwargs)

        try:
            client = await self._get_client()
            response = await client.post(self.endpoint_url, headers=headers, json=payload)
            latency_ms = (time.time() - start_time) * 1000
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self.provider_name, self.timeout)
        except httpx.RequestError as e:
            raise ProviderError(self.provider_name, f"Request error: {e}")

        if response.status_code == 429:
            await self.key_rotator.report_429(self.provider_name, slot_id)
            raise ProviderRateLimitError(self.provider_name, slot_id)

        if response.status_code == 400:
            body_text = response.text.lower()
            if any(kw in body_text for kw in ("context", "length", "token", "too long")):
                raise ProviderContextError(self.provider_name, model)
            raise ProviderError(self.provider_name, f"Bad request (400): {response.text[:300]}")

        if response.status_code in (401, 403):
            await self.key_rotator.report_429(self.provider_name, slot_id)
            raise ProviderAuthError(self.provider_name, slot_id)

        if response.status_code >= 500:
            raise ProviderError(self.provider_name, f"Server error ({response.status_code})")

        if response.status_code != 200:
            raise ProviderError(self.provider_name, f"Unexpected status {response.status_code}")

        await self.key_rotator.report_success(self.provider_name, slot_id)

        return self._parse_response(response.json(), latency_ms)


def _parse_tool_calls(raw_tool_calls: list[dict]) -> list[ToolCallResult]:
    """Parse tool calls - USE json.loads(), NOT eval()."""
    result = []
    for tc in raw_tool_calls:
        try:
            raw_args = tc.get("function", {}).get("arguments", "{}")
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                args = json.loads(raw_args)  # SAFE: json.loads, not eval
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse tool arguments: {e}")
            args = {}
        result.append(ToolCallResult(id=tc.get("id", ""), name=tc.get("function", {}).get("name", ""), arguments=args))
    return result


class GroqAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator):
        super().__init__("groq", key_rotator, timeout=30.0)

    @property
    def endpoint_url(self) -> str:
        return "https://api.groq.com/openai/v1/chat/completions"

    @property
    def default_model(self) -> str:
        return "llama-3.3-70b-versatile"

    def _build_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        payload = {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.3), "max_tokens": kwargs.get("max_tokens", 2048)}
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
            payload["tool_choice"] = "auto"
        return payload

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        choice = response_json["choices"][0]["message"]
        usage = response_json.get("usage", {})
        tool_calls = _parse_tool_calls(choice.get("tool_calls") or [])
        return LLMResponse(content=choice.get("content") or "", model=response_json.get("model", self.default_model), provider="groq", latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), tool_calls=tool_calls, raw_response=response_json)


class CerebrasAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator):
        super().__init__("cerebras", key_rotator, timeout=20.0)

    @property
    def endpoint_url(self) -> str:
        return "https://api.cerebras.ai/v1/chat/completions"

    @property
    def default_model(self) -> str:
        return "llama3.1-8b"

    def _build_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.3), "max_tokens": kwargs.get("max_tokens", 2048)}

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        choice = response_json["choices"][0]["message"]
        usage = response_json.get("usage", {})
        return LLMResponse(content=choice.get("content") or "", model=response_json.get("model", self.default_model), provider="cerebras", latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), raw_response=response_json)


class MistralAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator):
        super().__init__("mistral", key_rotator, timeout=60.0)

    @property
    def endpoint_url(self) -> str:
        return "https://api.mistral.ai/v1/chat/completions"

    @property
    def default_model(self) -> str:
        return "mistral-small-latest"

    def _build_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        payload = {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.3), "max_tokens": kwargs.get("max_tokens", 2048)}
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
            payload["tool_choice"] = "auto"
        return payload

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        choice = response_json["choices"][0]["message"]
        usage = response_json.get("usage", {})
        tool_calls = _parse_tool_calls(choice.get("tool_calls") or [])
        return LLMResponse(content=choice.get("content") or "", model=response_json.get("model", self.default_model), provider="mistral", latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), tool_calls=tool_calls, raw_response=response_json)


class NvidiaAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator):
        super().__init__("nvidia", key_rotator, timeout=30.0)

    @property
    def endpoint_url(self) -> str:
        return "https://integrate.api.nvidia.com/v1/chat/completions"

    @property
    def default_model(self) -> str:
        return "meta/llama-3.1-8b-instruct"

    def _build_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.3), "max_tokens": kwargs.get("max_tokens", 2048)}

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        choice = response_json["choices"][0]["message"]
        usage = response_json.get("usage", {})
        return LLMResponse(content=choice.get("content") or "", model=response_json.get("model", self.default_model), provider="nvidia", latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), raw_response=response_json)


class FireworksAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator):
        super().__init__("fireworks", key_rotator, timeout=60.0)

    @property
    def endpoint_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1/chat/completions"

    @property
    def default_model(self) -> str:
        return "accounts/fireworks/models/llama-v3p3-70b-instruct"

    def _build_headers(self, api_key: str) -> dict:
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "temperature": kwargs.get("temperature", 0.3), "max_tokens": kwargs.get("max_tokens", 2048)}

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        choice = response_json["choices"][0]["message"]
        usage = response_json.get("usage", {})
        return LLMResponse(content=choice.get("content") or "", model=response_json.get("model", self.default_model), provider="fireworks", latency_ms=latency_ms, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0), raw_response=response_json)


class OllamaAdapter(ProviderAdapter):
    def __init__(self, key_rotator: KeyRotator, base_host: str = "http://localhost:11434"):
        super().__init__("ollama", key_rotator, timeout=120.0)
        self._base_host = base_host

    @property
    def endpoint_url(self) -> str:
        return f"{self._base_host}/api/chat"

    @property
    def default_model(self) -> str:
        return "llama3.1"

    def _build_headers(self, api_key: str) -> dict:
        return {"Content-Type": "application/json"}

    def _build_payload(self, model: str, messages: list[dict], **kwargs) -> dict:
        return {"model": model, "messages": messages, "stream": False, "options": {"temperature": kwargs.get("temperature", 0.3), "num_predict": kwargs.get("max_tokens", 2048)}}

    def _parse_response(self, response_json: dict, latency_ms: float) -> LLMResponse:
        return LLMResponse(content=response_json.get("message", {}).get("content", ""), model=response_json.get("model", self.default_model), provider="ollama", latency_ms=latency_ms, raw_response=response_json)


ADAPTER_REGISTRY = {
    "groq": GroqAdapter,
    "cerebras": CerebrasAdapter,
    "mistral": MistralAdapter,
    "nvidia": NvidiaAdapter,
    "fireworks": FireworksAdapter,
    "ollama": OllamaAdapter,
}


def create_adapter(provider: str, key_rotator: KeyRotator) -> ProviderAdapter:
    if provider not in ADAPTER_REGISTRY:
        raise ValueError(f"Provider '{provider}' not supported. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[provider](key_rotator)


if __name__ == "__main__":
    async def smoke_test():
        from llm.key_rotator import KeyRotator, AllKeysExhaustedError
        rotator = KeyRotator()
        await rotator.register_keys("groq", ["test_key_fakexxx"])
        adapter = create_adapter("groq", rotator)
        print(f"PASS: GroqAdapter created - endpoint: {adapter.endpoint_url}")
        
        import inspect
        source = inspect.getsource(type(adapter))
        assert "eval(" not in source, "FAIL: eval() found!"
        print("PASS: ZERO eval() in adapter code")
        print("=== ALL TESTS PASSED ===")
    asyncio.run(smoke_test())
