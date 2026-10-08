"""Model gateway over OpenRouter: chat completions and embeddings, with cost, latency and retries."""

import os
import time
from typing import Any, Callable, Sequence

import httpx
from dotenv import load_dotenv

from agentic_jev_rag.gateway import Completion, Embeddings

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_VARIABLE = "OPEN_ROUTER_API_TOKEN"
TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


class OpenRouterError(Exception):
    def __init__(self, message: str, retries: Sequence[str] = ()):
        super().__init__(message)
        self.retries = tuple(retries)


class _Transient(Exception):
    def __init__(self, reason: str, wait_s: float | None = None):
        super().__init__(reason)
        self.wait_s = wait_s


class OpenRouterGateway:
    """Calls OpenRouter models. Latency covers the whole call, retries and backoff included."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        max_attempts: int = 4,
        backoff_s: float = 1.0,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "agentic-jev-rag"},
            timeout=timeout_s,
            transport=transport,
        )
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._clock = clock
        self._sleep = sleep

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenRouterGateway":
        load_dotenv()
        api_key = os.environ.get(API_KEY_VARIABLE)
        if not api_key:
            raise OpenRouterError(f"{API_KEY_VARIABLE} is not set; copy .env.example to .env")
        return cls(api_key, **kwargs)

    def complete(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        response_schema: tuple[str, dict] | None = None,
    ) -> Completion:
        """Chat completion. `response_schema` is (name, JSON schema) for strict structured output."""
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        request: dict[str, Any] = {"model": model, "messages": messages}
        if response_schema:
            name, schema = response_schema
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            }
        body, latency, retries = self._post("/chat/completions", request)
        return Completion(
            text=body["choices"][0]["message"]["content"],
            latency_s=latency,
            cost_usd=body["usage"]["cost"],
            retries=retries,
        )

    def embed(self, model: str, texts: Sequence[str]) -> Embeddings:
        body, latency, retries = self._post("/embeddings", {"model": model, "input": list(texts)})
        data = sorted(body["data"], key=lambda e: e["index"])
        return Embeddings(
            vectors=[e["embedding"] for e in data],
            latency_s=latency,
            cost_usd=body["usage"]["cost"],
            retries=retries,
        )

    def _post(self, path: str, request: dict) -> tuple[dict, float, tuple[str, ...]]:
        retries: list[str] = []
        start = self._clock()
        for attempt in range(1, self._max_attempts + 1):
            try:
                body = self._attempt(path, request)
                return body, self._clock() - start, tuple(retries)
            except _Transient as transient:
                retries.append(f"attempt {attempt}: {transient}")
                if attempt < self._max_attempts:
                    self._sleep(transient.wait_s or self._backoff_s * 2 ** (attempt - 1))
        raise OpenRouterError(
            f"{path} failed after {self._max_attempts} attempts: {retries[-1]}", retries
        )

    def _attempt(self, path: str, request: dict) -> dict:
        try:
            response = self._http.post(path, json=request)
        except httpx.TransportError as error:
            raise _Transient(f"{type(error).__name__}: {error}") from error

        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        if response.is_success and body is not None and not error:
            return body

        status = error.get("code", response.status_code) if error else response.status_code
        message = f"HTTP {status}: {error.get('message') if error else response.text[:200]}"
        if status in TRANSIENT_STATUSES:
            raise _Transient(message, _seconds(response.headers.get("Retry-After")))
        raise OpenRouterError(message)


def _seconds(retry_after: str | None) -> float | None:
    """Retry-After in seconds; the HTTP-date form is ignored in favour of normal backoff."""
    try:
        return float(retry_after) if retry_after else None
    except ValueError:
        return None
