"""Scripted fake gateways for RAG and Recruiter tests."""

import json

from agentic_jev_rag.gateway import FakeModelGateway


def scripted_gateway(
    *,
    route="candidates",
    verdict="allowed",
    relevant=None,
    grounded=True,
    answer="Ana built Kafka pipelines [1].",
    chat_responder=None,
    structured=None,
):
    """A fake whose System 1 decisions (LLM and Jev) and System 2 answers follow the arguments.

    `relevant(i)` decides chunk relevance by position (default: keep all). `structured` maps extra
    response-schema names to payloads (or callables taking the prompt) for agent completions.
    """
    keep = relevant or (lambda i: True)
    structured = structured or {}

    def complete(model, prompt, system, schema):
        if schema is None:
            return answer
        name = schema[0]
        if name == "route":
            return json.dumps({"collection": route})
        if name == "input_guardrail":
            return json.dumps({"verdict": verdict})
        if name == "chunk_relevance":
            count = len([line for line in prompt.splitlines() if line[:1] == "[" and "]" in line[:5]])
            return json.dumps({"relevant": [keep(i) for i in range(count)]})
        if name == "groundedness":
            return json.dumps({"grounded": grounded})
        payload = structured[name]
        return json.dumps(payload(prompt) if callable(payload) else payload)

    def decide(state, questions):
        answers = {}
        for qid, question in questions.items():
            if qid == "route":
                answers[qid] = {"type": "choice", "choice": route}
            elif qid == "verdict":
                answers[qid] = {"type": "choice", "choice": verdict}
            elif qid.startswith("chunk_"):
                answers[qid] = {"type": "noul", "noul": 0.9 if keep(int(qid.split("_")[1])) else 0.1}
            elif qid == "grounded":
                answers[qid] = {"type": "noul", "noul": 0.9 if grounded else 0.1}
        return answers

    return FakeModelGateway(complete_responder=complete, decide_responder=decide, chat_responder=chat_responder)


def turns(*replies):
    """A chat responder that returns the given replies in order (str or list of ToolCall), then 'Done.'."""
    queue = list(replies)

    def respond(model, messages, tools):
        return queue.pop(0) if queue else "Done."

    return respond
