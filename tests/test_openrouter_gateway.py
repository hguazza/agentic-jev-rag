"""Model gateway HTTP behaviour against recorded OpenRouter responses."""

import json
from pathlib import Path

import httpx
import pytest

from agentic_jev_rag.openrouter import OpenRouterError, OpenRouterGateway

RECORDED = Path(__file__).parent / "fixtures" / "openrouter"


def recorded(name: str) -> dict:
    return json.loads((RECORDED / f"{name}.json").read_text())


class FakeClock:
    def __init__(self, *ticks: float):
        self.ticks = list(ticks)

    def __call__(self) -> float:
        return self.ticks.pop(0)


def gateway_replying(*replies, clock=None, sleeps=None, max_attempts=3):
    """A gateway whose HTTP layer answers each request with the next reply and records requests."""
    requests: list[httpx.Request] = []
    replies = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    gateway = OpenRouterGateway(
        "test-key",
        transport=httpx.MockTransport(handle),
        clock=clock or FakeClock(*[float(i) for i in range(20)]),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
        max_attempts=max_attempts,
    )
    return gateway, requests


def test_completion_returns_text_reported_cost_and_measured_latency():
    gateway, _ = gateway_replying(
        httpx.Response(200, json=recorded("chat_completion")), clock=FakeClock(10.0, 10.75)
    )

    completion = gateway.complete("anthropic/claude-haiku-4.5", "Which court delivered Schrems?")

    assert completion.text == (
        "The Court of Justice of the European Union (CJEU) delivered the Schrems judgment."
    )
    assert completion.cost_usd == 0.000147
    assert completion.latency_s == 0.75
    assert completion.retries == ()


def test_completion_request_carries_model_messages_and_key():
    gateway, requests = gateway_replying(httpx.Response(200, json=recorded("chat_completion")))

    gateway.complete("anthropic/claude-haiku-4.5", "Question?", system="Answer briefly.")

    [request] = requests
    body = json.loads(request.content)
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "anthropic/claude-haiku-4.5"
    assert body["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Question?"},
    ]


ROUTE_SCHEMA = {
    "type": "object",
    "properties": {"collection": {"type": "string", "enum": ["case_law", "legislation"]}},
    "required": ["collection"],
    "additionalProperties": False,
}


def test_structured_completion_requests_a_strict_json_schema_and_returns_json_text():
    gateway, requests = gateway_replying(
        httpx.Response(200, json=recorded("chat_completion_structured"))
    )

    completion = gateway.complete(
        "anthropic/claude-haiku-4.5", "Which Collection?", response_schema=("route", ROUTE_SCHEMA)
    )

    assert json.loads(completion.text) == {"collection": "legislation"}
    assert json.loads(requests[0].content)["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "route", "strict": True, "schema": ROUTE_SCHEMA},
    }


def rate_limited() -> httpx.Response:
    # Constructed in OpenRouter's documented error shape; transient errors can't be recorded on demand.
    return httpx.Response(429, json={"error": {"message": "Rate limit exceeded", "code": 429}})


def test_transient_errors_are_retried_with_backoff_and_recorded():
    sleeps: list[float] = []
    gateway, requests = gateway_replying(
        rate_limited(),
        httpx.ConnectTimeout("timed out"),
        httpx.Response(200, json=recorded("chat_completion")),
        sleeps=sleeps,
    )

    completion = gateway.complete("anthropic/claude-haiku-4.5", "Question?")

    assert len(requests) == 3
    assert sleeps == [1.0, 2.0]
    assert len(completion.retries) == 2
    assert "429" in completion.retries[0]
    assert "timed out" in completion.retries[1]


def test_retry_after_header_sets_the_wait():
    sleeps: list[float] = []
    gateway, _ = gateway_replying(
        httpx.Response(503, headers={"Retry-After": "7"}, json={"error": {"message": "busy", "code": 503}}),
        httpx.Response(200, json=recorded("chat_completion")),
        sleeps=sleeps,
    )

    gateway.complete("anthropic/claude-haiku-4.5", "Question?")

    assert sleeps == [7.0]


def test_gives_up_after_max_attempts_and_reports_every_retry():
    gateway, requests = gateway_replying(rate_limited(), rate_limited(), rate_limited(), max_attempts=3)

    with pytest.raises(OpenRouterError, match="3 attempts") as error:
        gateway.complete("anthropic/claude-haiku-4.5", "Question?")

    assert len(requests) == 3
    assert len(error.value.retries) == 3


def test_non_transient_errors_fail_at_once_with_openrouters_message():
    gateway, requests = gateway_replying(
        httpx.Response(400, json=recorded("error_invalid_model"))
    )

    with pytest.raises(OpenRouterError, match="not a valid model ID"):
        gateway.complete("nonexistent/model-xyz", "Question?")

    assert len(requests) == 1


def test_embeddings_return_one_vector_per_input_in_order_with_cost():
    reply = recorded("embeddings")
    reply["data"].reverse()  # OpenRouter tags each vector with its input index; order is not guaranteed
    gateway, requests = gateway_replying(httpx.Response(200, json=reply), clock=FakeClock(1.0, 1.5))
    texts = ["transfer of personal data to third countries", "one-stop shop mechanism"]

    embeddings = gateway.embed("baai/bge-m3", texts)

    expected = {e["index"]: e["embedding"] for e in recorded("embeddings")["data"]}
    assert embeddings.vectors == [expected[0], expected[1]]
    assert embeddings.cost_usd == 1.6e-07
    assert embeddings.latency_s == 0.5
    assert requests[0].url == "https://openrouter.ai/api/v1/embeddings"
    assert json.loads(requests[0].content) == {"model": "baai/bge-m3", "input": texts}


def test_non_json_gateway_errors_are_retried():
    gateway, requests = gateway_replying(
        httpx.Response(502, text="<html>Bad Gateway</html>", headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        httpx.Response(200, json=recorded("chat_completion")),
    )

    completion = gateway.complete("anthropic/claude-haiku-4.5", "Question?")

    assert len(requests) == 2
    assert "502" in completion.retries[0]
