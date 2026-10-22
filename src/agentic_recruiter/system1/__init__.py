"""System 1 decisions behind one interface, made by the LLM decider or the Jev decider."""

from agentic_recruiter.config import Config
from agentic_recruiter.gateway import ModelGateway
from agentic_recruiter.system1.jev import JevDecider
from agentic_recruiter.system1.llm import LlmDecider
from agentic_recruiter.system1.types import (
    GroundednessDecision,
    GuardrailDecision,
    RelevanceDecision,
    RouteDecision,
    SignalsDecision,
    System1Decider,
    System1Error,
    System1TraceEntry,
    TurnCheckDecision,
)

__all__ = [
    "GroundednessDecision", "GuardrailDecision", "JevDecider", "LlmDecider", "RelevanceDecision",
    "RouteDecision", "SignalsDecision", "System1Decider", "System1Error", "System1TraceEntry", "TurnCheckDecision",
    "make_deciders",
]


def make_deciders(config: Config, gateway: ModelGateway) -> dict[str, System1Decider]:
    return {
        "llm": LlmDecider(gateway, config.models.system_1_baseline),
        "jev": JevDecider(gateway, config.models.jev_model),
    }
