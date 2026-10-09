"""The fake Model gateway used by every offline test."""

from agentic_jev_rag.gateway import FakeModelGateway, ToolCall


def test_scripted_chat_returns_tool_calls():
    gateway = FakeModelGateway(
        chat_responder=lambda model, messages, tools: [ToolCall("c1", "search", {"query": "x"})]
    )

    result = gateway.chat("m", [{"role": "user", "content": "hi"}], tools=[])

    assert result.tool_calls == (ToolCall("c1", "search", {"query": "x"}),)
    assert result.content is None


def test_default_decisions_answer_every_question_type():
    gateway = FakeModelGateway()

    answers = gateway.decide(
        "jev",
        "state",
        {
            "a": {"type": "noul", "instructions": "?"},
            "b": {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}},
            "c": {"type": "score", "instructions": "?", "criteria": ["low", "high"]},
        },
    ).answers

    assert answers["a"]["noul"] == 0.0
    assert answers["b"]["choice"] == "x"
    assert answers["c"]["score"] == 0


def test_fake_rerank_prefers_documents_sharing_words_with_the_query():
    scores = FakeModelGateway().rerank("r", "python kafka", ["excel tableau", "python and kafka"]).scores

    assert scores[1] > scores[0]


def test_calls_are_recorded_by_kind():
    gateway = FakeModelGateway()
    gateway.complete("m", "prompt")
    gateway.embed("e", ["text"])

    assert [kind for kind, _, _ in gateway.calls] == ["complete", "embed"]
