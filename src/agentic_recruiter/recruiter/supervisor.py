"""The Supervisor: the System 2 model picks the next agent; invariants keep the order safe."""

from dataclasses import dataclass

from agentic_recruiter.recruiter import prompts
from agentic_recruiter.recruiter.context import RecruiterContext

HANDOFFS = {
    "to_sourcer": "sourcer",
    "to_screener": "screener",
    "to_interviewer": "interviewer",
    "to_compliance": "compliance",
    "finish": "finish",
}


@dataclass(frozen=True)
class Progress:
    goal: str
    job_id: str
    sourced: bool = False
    shortlisted: int = 0
    screened: int = 0
    advanced: int = 0
    interviewed: int = 0
    compliance_done: bool = False
    interviews_enabled: bool = True

    def pending(self) -> list[str]:
        """Steps that have work left and whose prerequisites are met, in the required order."""
        if not self.sourced:
            return ["sourcer"]
        steps = []
        if self.shortlisted > self.screened:
            steps.append("screener")
        if self.interviews_enabled and self.advanced > self.interviewed:
            steps.append("interviewer")
        if steps:
            return steps
        return ["finish"] if self.compliance_done else ["compliance"]


@dataclass(frozen=True)
class SupervisorDecision:
    choice: str
    corrected_from: str | None
    cost_usd: float
    latency_s: float


def supervise(ctx: RecruiterContext, progress: Progress) -> SupervisorDecision:
    tools = [
        {"type": "function", "function": {"name": name, "description": f"Hand off: {step}", "parameters": {"type": "object", "properties": {}}}}
        for name, step in HANDOFFS.items()
    ]
    status = (
        f"Goal: {progress.goal}\nJob: {progress.job_id}\nSourced: {progress.sourced}\n"
        f"Shortlisted: {progress.shortlisted}\nScreened: {progress.screened}\nAdvanced by the Screener: {progress.advanced}\n"
        f"Interviewed: {progress.interviewed} (interviews enabled: {progress.interviews_enabled})\n"
        f"Compliance review done: {progress.compliance_done}"
    )
    turn = ctx.gateway.chat(
        ctx.config.models.system_2_model,
        [{"role": "system", "content": prompts.SUPERVISOR}, {"role": "user", "content": status}],
        tools=tools,
    )
    proposed = HANDOFFS.get(turn.tool_calls[0].name) if turn.tool_calls else None
    pending = progress.pending()
    choice = proposed if proposed in pending else pending[0]
    return SupervisorDecision(choice, None if choice == proposed else proposed or "none", turn.cost_usd, turn.latency_s)
