"""The tool-calling agent loop and the RAG tools agents use."""

from datetime import date

from fastapi.testclient import TestClient

from agentic_recruiter.gateway import ChatResult, FakeModelGateway, ToolCall
from agentic_recruiter.rag.api import create_rag_app
from agentic_recruiter.rag.service import RagService
from agentic_recruiter.recruiter.agent_loop import Tool, run_agent
from agentic_recruiter.recruiter.rag_client import RagClient, RagError
from agentic_recruiter.recruiter.tools import rag_tools
from tests.rag_scripts import scripted_gateway, turns

ECHO = Tool("echo", "Echo text back", {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}, lambda text: {"echo": text})
SUBMIT = Tool("submit_answer", "Submit the final answer", {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}, lambda **kw: kw)


def test_agent_uses_a_tool_then_submits():
    gateway = FakeModelGateway(chat_responder=turns(
        [ToolCall("1", "echo", {"text": "hi"})],
        [ToolCall("2", "submit_answer", {"answer": "hi back"})],
    ))

    result = run_agent(gateway, "m", "You echo.", "Say hi.", [ECHO], SUBMIT)

    assert result.submitted == {"answer": "hi back"}
    assert result.stopped_reason == "submitted"
    assert result.steps == 2
    tool_message = next(m for m in result.transcript if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "1" and '"echo": "hi"' in tool_message["content"]


def test_costs_and_latency_are_summed_over_turns():
    class Priced(FakeModelGateway):
        def chat(self, model, messages, *, tools=None):
            r = super().chat(model, messages, tools=tools)
            return ChatResult(r.content, r.tool_calls, latency_s=0.5, cost_usd=0.001)

    gateway = Priced(chat_responder=turns([ToolCall("1", "echo", {"text": "x"})], [ToolCall("2", "submit_answer", {"answer": "x"})]))

    result = run_agent(gateway, "m", "s", "t", [ECHO], SUBMIT)

    assert result.cost_usd == 0.002
    assert result.latency_s == 1.0


def test_an_agent_that_never_submits_stops_at_the_step_limit():
    gateway = FakeModelGateway(chat_responder=lambda m, msgs, tools: [ToolCall(str(len(msgs)), "echo", {"text": "again"})])

    result = run_agent(gateway, "m", "s", "t", [ECHO], SUBMIT, max_steps=3)

    assert result.submitted is None
    assert result.stopped_reason == "step_limit"
    assert result.steps == 3


def test_an_agent_that_only_chats_is_nudged_once_then_stopped():
    gateway = FakeModelGateway(chat_responder=lambda m, msgs, tools: "I think I'm done.")

    result = run_agent(gateway, "m", "s", "t", [ECHO], SUBMIT)

    assert result.stopped_reason == "no_tool_call"
    assert result.steps == 2


def test_unknown_tools_and_tool_errors_are_reported_back_to_the_model():
    def boom(text):
        raise ValueError("bad input")

    failing = Tool("echo", "Echo", ECHO.parameters, boom)
    gateway = FakeModelGateway(chat_responder=turns(
        [ToolCall("1", "nope", {}), ToolCall("2", "echo", {"text": "x"})],
        [ToolCall("3", "submit_answer", {"answer": "ok"})],
    ))

    result = run_agent(gateway, "m", "s", "t", [failing], SUBMIT)

    contents = [m["content"] for m in result.transcript if m["role"] == "tool"]
    assert "unknown tool" in contents[0]
    assert "bad input" in contents[1]
    assert result.submitted == {"answer": "ok"}


def test_a_submit_missing_required_fields_is_rejected_and_the_agent_can_retry():
    gateway = FakeModelGateway(chat_responder=turns(
        [ToolCall("1", "submit_answer", {})],
        [ToolCall("2", "submit_answer", {"answer": "fixed"})],
    ))

    result = run_agent(gateway, "m", "s", "t", [], SUBMIT)

    assert result.submitted == {"answer": "fixed"}
    assert "answer" in next(m for m in result.transcript if m["role"] == "tool")["content"]


def test_tools_are_offered_in_openai_format():
    gateway = FakeModelGateway(chat_responder=turns([ToolCall("1", "submit_answer", {"answer": "a"})]))

    run_agent(gateway, "m", "s", "t", [ECHO], SUBMIT)

    [(_, _, payload)] = gateway.calls
    names = [t["function"]["name"] for t in payload["tools"]]
    assert names == ["echo", "submit_answer"]
    assert payload["tools"][0]["type"] == "function"


def test_rag_client_and_tools_call_the_rag_api(indexed, config, qdrant, fixture_documents):
    service = RagService(config, scripted_gateway(), qdrant, fixture_documents, today=lambda: date(2026, 10, 8))
    rag = RagClient("http://rag", http=TestClient(create_rag_app(service)))
    tools = rag_tools(rag)

    matched = tools["match_candidates"].fn(job_id="J-001", top_k=2)
    document = tools["get_document"].fn(document_id="J-001")

    assert len(matched["candidates"]) == 2
    assert document["id"] == "J-001"
    assert set(tools) == {"search", "match_candidates", "ask", "get_document"}
    try:
        rag.document("NOPE")
        raise AssertionError("expected RagError")
    except RagError as error:
        assert "404" in str(error)


def test_a_submission_rejected_by_the_submit_tools_validator_is_sent_back_for_correction():
    def validate(answer):
        if not answer.strip():
            raise ValueError("answer must not be blank")
        return {"answer": answer.upper()}

    strict = Tool("submit_answer", "Submit", SUBMIT.parameters, validate)
    gateway = FakeModelGateway(chat_responder=turns(
        [ToolCall("1", "submit_answer", {"answer": " "})],
        [ToolCall("2", "submit_answer", {"answer": "ok"})],
    ))

    result = run_agent(gateway, "m", "s", "t", [], strict)

    assert result.submitted == {"answer": "OK"}
    assert "must not be blank" in next(m for m in result.transcript if m["role"] == "tool")["content"]


def test_long_tool_results_are_cut_with_a_visible_marker():
    big = Tool("big", "Big", {"type": "object", "properties": {}}, lambda: "x" * 50_000)
    gateway = FakeModelGateway(chat_responder=turns([ToolCall("1", "big", {})], [ToolCall("2", "submit_answer", {"answer": "a"})]))

    result = run_agent(gateway, "m", "s", "t", [big], SUBMIT)

    content = next(m for m in result.transcript if m["role"] == "tool")["content"]
    assert content.endswith("[truncated]")


def test_rag_tools_return_compact_match_results_that_fit_without_truncation(indexed, config, qdrant, fixture_documents):
    from agentic_recruiter.recruiter.agent_loop import TOOL_RESULT_LIMIT
    import json as _json

    service = RagService(config, scripted_gateway(), qdrant, fixture_documents, today=lambda: date(2026, 10, 8))
    tools = rag_tools(RagClient("http://rag", http=TestClient(create_rag_app(service))))

    matched = tools["match_candidates"].fn(job_id="J-001", top_k=5)

    assert len(_json.dumps(matched)) < TOOL_RESULT_LIMIT
    evidence = matched["candidates"][0]["evidence"][0]
    assert set(evidence) == {"chunk_id", "section", "text"}


def test_an_empty_assistant_turn_is_sent_back_with_empty_text_not_null():
    assert ChatResult(None, (), 0.0, 0.0).as_message() == {"role": "assistant", "content": ""}
