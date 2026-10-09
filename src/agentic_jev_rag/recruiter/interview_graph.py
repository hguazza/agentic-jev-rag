"""A live interview as a LangGraph graph that pauses for each Candidate reply (interrupt + checkpointer)."""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agentic_jev_rag.recruiter import interviewer
from agentic_jev_rag.recruiter.context import RecruiterContext
from agentic_jev_rag.recruiter.schemas import InterviewAssessment, InterviewTurn


class InterviewGraphState(TypedDict, total=False):
    candidate: dict
    job: dict
    turns: list[InterviewTurn]
    question: str | None
    assessment: InterviewAssessment | None


def build_interview_graph(ctx: RecruiterContext, *, max_questions: int = 4):
    # Asking and waiting are separate nodes: LangGraph re-runs an interrupted node on resume,
    # and the question must not be regenerated.
    def ask(state: InterviewGraphState) -> dict:
        turns = state.get("turns", [])
        return {"question": interviewer.next_question(ctx, state["candidate"], state["job"], turns, max_questions)}

    def wait_for_answer(state: InterviewGraphState) -> dict:
        answer = interrupt({"question": state["question"]})
        return {"turns": state.get("turns", []) + [InterviewTurn(question=state["question"], answer=answer)]}

    def assess(state: InterviewGraphState) -> dict:
        return {"assessment": interviewer.assess(ctx, state["candidate"], state["job"], state.get("turns", [])), "question": None}

    graph = StateGraph(InterviewGraphState)
    graph.add_node("ask", ask)
    graph.add_node("wait_for_answer", wait_for_answer)
    graph.add_node("assess", assess)
    graph.add_edge(START, "ask")
    graph.add_conditional_edges("ask", lambda s: "assess" if s["question"] is None else "wait_for_answer")
    graph.add_edge("wait_for_answer", "ask")
    graph.add_edge("assess", END)
    return graph.compile(checkpointer=InMemorySaver())
