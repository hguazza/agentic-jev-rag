"""Scripted fake gateways for RAG and Recruiter tests."""

import json

from agentic_recruiter.gateway import FakeModelGateway


TURN_OUTCOMES = {
    "substantive": {"intelligibility": "answer", "injection": 0.0, "depth": "substantive"},
    "shallow": {"intelligibility": "answer", "injection": 0.0, "depth": "shallow"},
    "unintelligible": {"intelligibility": "unintelligible", "injection": 0.0, "depth": "shallow"},
    "off_topic": {"intelligibility": "off_topic", "injection": 0.0, "depth": "shallow"},
    "injection": {"intelligibility": "answer", "injection": 0.9, "depth": "shallow"},
}
ANSWER_MARK = "Answer (speech-to-text transcript): "


def scripted_gateway(
    *,
    route="candidates",
    verdict="allowed",
    relevant=None,
    grounded=True,
    answer="Ana built Kafka pipelines [1].",
    chat_responder=None,
    structured=None,
    turn=None,
    consistency="consistent",
):
    """A fake whose System 1 decisions (LLM and Jev) and System 2 answers follow the arguments.

    `relevant(i)` decides chunk relevance by position (default: keep all). `structured` maps extra
    response-schema names to payloads (or callables taking the prompt) for agent completions.
    `turn` is a Turn check outcome dict, or a callable from the answer to one; None from the callable makes the decision fail.
    `consistency` is the Interview signals label, or a callable from the answer to one.
    """
    keep = relevant or (lambda i: True)
    structured = structured or {}

    consistency_for = consistency if callable(consistency) else (lambda answer: consistency)
    turn_for = turn if callable(turn) else (lambda answer: turn or TURN_OUTCOMES["substantive"])

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
        if name == "turn_check":
            outcome = turn_for(prompt.split(ANSWER_MARK, 1)[1].split("\n\n", 1)[0])
            if outcome is None:
                return "not json"
            return json.dumps({**outcome, "injection": outcome["injection"] >= 0.5})
        if name == "interview_signals":
            answer_text = prompt.split("Candidate's interview answer: ", 1)[1].split("\n\n", 1)[0]
            return json.dumps({"evidence": [True] * 8, "consistency": consistency_for(answer_text)})
        payload = structured[name]
        return json.dumps(payload(prompt) if callable(payload) else payload)

    def decide(state, questions):
        outcome = turn_for(state["answer"]) if "intelligibility" in questions else None
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
            elif qid == "intelligibility" and outcome:
                answers[qid] = {"type": "choice", "choice": outcome["intelligibility"]}
            elif qid == "injection" and outcome:
                answers[qid] = {"type": "noul", "noul": outcome["injection"]}
            elif qid == "depth" and outcome:
                answers[qid] = {"type": "choice", "choice": outcome["depth"]}
            elif qid.startswith("competency_"):
                answers[qid] = {"type": "noul", "noul": 0.9}
            elif qid == "consistency":
                answers[qid] = {"type": "choice", "choice": consistency_for(state["answer"])}
        return answers

    return FakeModelGateway(complete_responder=complete, decide_responder=decide, chat_responder=chat_responder)


def turns(*replies):
    """A chat responder that returns the given replies in order (str or list of ToolCall), then 'Done.'."""
    queue = list(replies)

    def respond(model, messages, tools):
        return queue.pop(0) if queue else "Done."

    return respond


def by_agent(**responders):
    """A chat responder that dispatches on which agent is talking (the name in its system prompt)."""

    def respond(model, messages, tools):
        system = messages[0]["content"]
        for name, responder in responders.items():
            if system.startswith(f"You are the {name}"):
                return responder(model, messages, tools)
        raise AssertionError(f"no scripted responder for system prompt: {system[:60]!r}")

    return respond
