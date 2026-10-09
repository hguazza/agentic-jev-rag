"""What the agents produce: Shortlists, Screening verdicts, Interview assessments and Compliance findings.

These models are both the Recruiter API's response types and the schemas of the agents' submit tools, so
their descriptions are read by people (in `/docs`) and by the models (in the tool definitions).
"""

from typing import Literal

from pydantic import BaseModel, Field


class ShortlistEntry(BaseModel):
    """A Candidate the Sourcer proposes for the Job."""

    candidate_id: str = Field(description="The Candidate's id, as returned by Match or Search.", examples=["C-0040"])
    name: str = Field(description="The Candidate's name.", examples=["Lucas Singh"])
    reason: str = Field(description="A job-related reason for shortlisting, referring to the Candidate's evidence.")


class Shortlist(BaseModel):
    """The Candidates the Sourcer proposes for a Job, best first. Candidates without valid Consent are removed."""

    job_id: str = Field(description="The Job the shortlist is for.", examples=["J-037"])
    candidates: list[ShortlistEntry] = Field(
        default_factory=list, max_length=8, description="At most 8 Candidates, best first."
    )


class CompetencyScore(BaseModel):
    """A score for one Scorecard competency."""

    competency: str = Field(description="The Scorecard competency, as named in the Scorecards policy.", examples=["Technical operations"])
    score: int = Field(ge=1, le=5, description="1 (no evidence) to 5 (strong, repeated evidence at the required seniority).")
    evidence_chunk_ids: list[str] = Field(
        default_factory=list, description="Ids of the chunks (`<document_id>#<n>`) that support the score.", examples=[["C-0040#2"]]
    )
    rationale: str = Field(description="Why this score, quoting or paraphrasing the evidence.")


class ScreeningVerdict(BaseModel):
    """The Screener's assessment of one Candidate against a Job's Scorecard."""

    candidate_id: str = Field(description="The Candidate screened.", examples=["C-0040"])
    job_id: str = Field(description="The Job screened for.", examples=["J-037"])
    decision: Literal["advance", "reject"] = Field(
        description="`advance` when no competency scores 1 and the average is 3 or more, otherwise `reject`."
    )
    scores: list[CompetencyScore] = Field(min_length=1, description="One score per Scorecard competency.")
    summary: str = Field(description="A short, job-related summary of the verdict.")


class InterviewTurn(BaseModel):
    """One question and the Candidate's answer."""

    question: str = Field(description="The Interviewer's question.")
    answer: str = Field(description="The Candidate's answer (or the Simulated candidate's, in autonomous runs).")


class InterviewAssessment(BaseModel):
    """The Interviewer's assessment after an interview."""

    candidate_id: str = Field(description="The Candidate interviewed.", examples=["C-0040"])
    job_id: str = Field(description="The Job interviewed for.", examples=["J-037"])
    turns: list[InterviewTurn] = Field(description="The interview transcript, in order.")
    scores: list[CompetencyScore] = Field(description="One score per Scorecard competency, with rationales quoting the answers.")
    recommendation: Literal["hire", "no_hire", "next_round"] = Field(
        description="`hire` when every competency is 3 or more and the average 4 or more; `next_round` when the "
        "average is between 3 and 4; otherwise `no_hire`. A human recruiter reviews every recommendation."
    )
    summary: str = Field(description="A short summary of the interview.")


class ComplianceFinding(BaseModel):
    """The result of one Compliance check."""

    check: Literal["input_guardrail", "consent", "groundedness", "policy"] = Field(
        description="`input_guardrail`: the goal was screened for discrimination and scope. `consent`: a shortlisted "
        "Candidate's Consent. `groundedness`: whether a Screening verdict is supported by its evidence. `policy`: a "
        "policy requirement that applies to the results."
    )
    subject: str = Field(description="What was checked: `goal`, a Candidate id, or a policy topic.", examples=["C-0040"])
    passed: bool = Field(description="Whether the check passed. A failed check needs human attention.")
    detail: str = Field(description="An explanation of the result.")


class AgentRunNote(BaseModel):
    """How one agent run (or a Supervisor correction) ended."""

    agent: str = Field(description="The agent: `sourcer`, `screener` or `supervisor`.", examples=["screener"])
    subject: str = Field(default="", description="What the agent worked on: a Job id, a Candidate id, or a correction.")
    stopped_reason: str = Field(
        description="`submitted` (finished normally), `step_limit`, `no_tool_call`, or `corrected` for a Supervisor "
        "hand-off that was out of order."
    )
    steps: int = Field(description="Model turns used.")
    cost_usd: float = Field(description="Model cost of the run reported by OpenRouter, in US dollars.")


# Descriptions are kept: they tell the model what each field means.
UNSUPPORTED_KEYWORDS = {"title", "default", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "examples"}


def json_schema(model: type[BaseModel], *, exclude: tuple[str, ...] = ()) -> dict:
    """A model's JSON schema made strict-friendly: references inlined, every field required, no extra
    properties, and range keywords dropped (Pydantic still enforces them when the result is validated).
    Fields the caller fills in itself are excluded."""
    raw = model.model_json_schema()
    definitions = raw.pop("$defs", {})

    def clean(node):
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return clean(definitions[node["$ref"].split("/")[-1]])
        node = {k: clean(v) for k, v in node.items() if k not in UNSUPPORTED_KEYWORDS}
        if node.get("type") == "object" and "properties" in node:
            node["required"] = list(node["properties"])
            node["additionalProperties"] = False
        return node

    schema = clean(raw)
    for name in exclude:
        schema["properties"].pop(name, None)
    schema["required"] = [name for name in schema["required"] if name not in exclude]
    return schema
