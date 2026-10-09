"""Scripted interviews for the interview graph, the Recruiter API and the voice worker."""

import re

from langgraph.types import Command

from tests.rag_scripts import scripted_gateway

ASSESSMENT = {
    "scores": [{"competency": "System design", "score": 4, "evidence_chunk_ids": [], "rationale": "Partitioned topics by order id."}],
    "recommendation": "next_round",
    "summary": "Good interview.",
}


def interview_question(prompt):
    """Two questions, then done."""
    asked = int(re.search(r"Questions asked: (\d+)", prompt).group(1))
    return {"done": asked >= 2, "question": f"Tell me about question {asked + 1}."}


def interview_gateway(**kwargs):
    """A two-question interview. `structured` entries override the defaults; other kwargs go to scripted_gateway."""
    structured = {"interview_question": interview_question, "interview_assessment": ASSESSMENT} | kwargs.pop("structured", {})
    return scripted_gateway(structured=structured, **kwargs)


def thread(name="t"):
    return {"configurable": {"thread_id": name}}


def start(graph, ctx, candidate_id="C-001", job_id="J-001", name="t"):
    return graph.invoke({"candidate": ctx.rag.document(candidate_id), "job": ctx.rag.document(job_id), "turns": []}, thread(name))


def reply(graph, answer, name="t"):
    return graph.invoke(Command(resume=answer), thread(name))


def pending(state):
    """The interrupt payload the graph is waiting on, or None when it finished."""
    interrupts = state.get("__interrupt__")
    return interrupts[0].value if interrupts else None


def prompts_for(gateway, schema_name):
    """Prompts of the structured completions made with this response-schema name, in order."""
    return [p["prompt"] for kind, _, p in gateway.calls if kind == "complete" and p["schema"] and p["schema"][0] == schema_name]
