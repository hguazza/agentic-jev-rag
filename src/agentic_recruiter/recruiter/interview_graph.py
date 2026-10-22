"""A live interview as a LangGraph graph that pauses for each Candidate reply (interrupt + a checkpointer: in memory locally, DynamoDB on AWS).

Every reply gets a System 1 Turn check before the System 2 model is asked anything: unusable replies are
reprompted and prompt injections redirected without a System 2 call (at most `max_reprompts` per question).
Shallow answers get up to `max_probes_per_topic` forensic follow-ups, which do not count as topics.
Interview signals for each scored answer run in the same superstep as the next question or probe, so they cost no latency.
"""

from dataclasses import asdict
from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentic_recruiter.rag.models import TraceOut
from agentic_recruiter.recruiter import interviewer
from agentic_recruiter.recruiter.context import RecruiterContext
from agentic_recruiter.recruiter.integrity import integrity_signals
from agentic_recruiter.recruiter.schemas import InterviewAssessment, InterviewTurn, RedFlag, TurnTiming

UNUSABLE = ("unintelligible", "off_topic", "injection")


class InterviewGraphState(TypedDict, total=False):
    candidate: dict
    job: dict
    briefing: str
    competencies: list[str]
    dossier: str
    turns: list[InterviewTurn]
    question: str | None
    question_kind: str  # how the pending question is put: question, probe, reprompt or redirect
    competency: str  # of the pending question
    probes: int  # probes asked on the current topic
    topics: int  # new questions asked so far; probes, reprompts and redirects do not count
    retries: int  # reprompts and redirects for the pending question
    check: str  # the last Turn check label
    red_flags: list[RedFlag]
    signals: list[dict]  # Interview signals per scored answer
    trace: list[dict]  # System 1 trace entries, as dicts
    assessment: InterviewAssessment | None


def build_interview_graph(
    ctx: RecruiterContext, *, max_questions: int = 4, checkpointer: BaseCheckpointSaver | None = None
):
    budgets = ctx.config.voice

    def brief(state: InterviewGraphState) -> dict:
        briefing = interviewer.brief(ctx, state["candidate"], state["job"])
        return {"briefing": briefing.text, "competencies": list(briefing.competencies), "dossier": briefing.dossier}

    # Asking and waiting are separate nodes: LangGraph re-runs an interrupted node on resume,
    # and the question must not be regenerated.
    def ask(state: InterviewGraphState) -> dict:
        topics = state.get("topics", 0)
        question = interviewer.next_question(ctx, state["briefing"], _scored(state), max_questions, asked=topics)
        if question is None:
            return {"question": None}
        return {"question": question.text, "competency": question.competency, "question_kind": "question",
                "topics": topics + 1, "probes": 0, "retries": 0}

    def wait_for_answer(state: InterviewGraphState) -> dict:
        kind = state.get("question_kind", "question")
        reply = interrupt({"question": state["question"], "kind": kind})
        message, timing = (reply["message"], reply.get("timing")) if isinstance(reply, dict) else (reply, None)
        turn = InterviewTurn(
            question=state["question"], answer=message, kind=kind, competency=state.get("competency", ""),
            timing=TurnTiming(**timing) if timing else None,
        )
        return {"turns": state.get("turns", []) + [turn]}

    def turn_check(state: InterviewGraphState) -> dict:
        *earlier, last = state["turns"]
        decision = ctx.decider.turn_check(last.question, last.answer)
        checked = last.model_copy(update={"check": decision.label, "scored": decision.label not in UNUSABLE})
        update = {"turns": [*earlier, checked], "check": decision.label, "trace": state.get("trace", []) + [asdict(decision.trace)]}
        if decision.label == "injection":
            flag = RedFlag(kind="injection", quote=last.answer, detail="Tried to instruct the AI interviewer instead of answering; not scored.")
            update["red_flags"] = state.get("red_flags", []) + [flag]
        return update

    def after_check(state: InterviewGraphState) -> str | list[str]:
        if state["check"] in UNUSABLE and state.get("retries", 0) < budgets.max_reprompts:
            return "redirect" if state["check"] == "injection" else "reprompt"
        if state["check"] == "shallow" and state.get("probes", 0) < budgets.max_probes_per_topic:
            return ["probe", "observe"]
        return ["ask", "observe"]

    # Branches in one superstep (ask or probe, and observe) feed route_next and must write disjoint state keys.
    def observe(state: InterviewGraphState) -> dict:
        last = state["turns"][-1]
        if not last.scored:
            return {}
        decision = ctx.decider.interview_signals(last.answer, state.get("competencies", []), state["dossier"])
        signal = {"turn": len(state["turns"]), "answer": last.answer, "evidence": decision.evidence, "consistency": decision.consistency}
        return {"signals": state.get("signals", []) + [signal], "trace": state.get("trace", []) + [asdict(decision.trace)]}

    def route_next(state: InterviewGraphState) -> dict:
        return {}  # the fan-in point: runs once both branches of the superstep have written their updates

    def probe(state: InterviewGraphState) -> dict:
        question = interviewer.next_probe(ctx, state["briefing"], _scored(state))
        return {"question": question, "question_kind": "probe", "probes": state.get("probes", 0) + 1, "retries": 0}

    def reprompt(state: InterviewGraphState) -> dict:
        return {"question_kind": "reprompt", "retries": state.get("retries", 0) + 1}

    def redirect(state: InterviewGraphState) -> dict:
        return {"question_kind": "redirect", "retries": state.get("retries", 0) + 1}

    def assess(state: InterviewGraphState) -> dict:
        assessment = interviewer.assess(
            ctx, state["briefing"], state["candidate"]["id"], state["job"]["id"], _scored(state), red_flags=state.get("red_flags", []),
            signals=state.get("signals", []),
        )
        full = assessment.model_copy(
            update={"turns": state.get("turns", []), "system_1_trace": [TraceOut(**t) for t in state.get("trace", [])],
                    "probes_sustained": interviewer.probes_sustained(state.get("turns", [])),
                    "integrity_signals": integrity_signals(state.get("turns", []))}
        )
        return {"assessment": full, "question": None}

    graph = StateGraph(InterviewGraphState)
    for name, node in [
        ("brief", brief), ("ask", ask), ("wait_for_answer", wait_for_answer), ("turn_check", turn_check),
        ("reprompt", reprompt), ("redirect", redirect), ("probe", probe), ("assess", assess),
        ("observe", observe), ("route_next", route_next),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "brief")
    graph.add_edge("brief", "ask")
    graph.add_edge("wait_for_answer", "turn_check")
    graph.add_conditional_edges("turn_check", after_check, ["reprompt", "redirect", "probe", "ask", "observe"])
    graph.add_edge("reprompt", "wait_for_answer")
    graph.add_edge("redirect", "wait_for_answer")
    for node in ("ask", "probe", "observe"):
        graph.add_edge(node, "route_next")
    graph.add_conditional_edges(
        "route_next", lambda s: "assess" if s["question"] is None else "wait_for_answer", ["assess", "wait_for_answer"]
    )
    graph.add_edge("assess", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def _scored(state: InterviewGraphState) -> list[InterviewTurn]:
    return [t for t in state.get("turns", []) if t.scored]
