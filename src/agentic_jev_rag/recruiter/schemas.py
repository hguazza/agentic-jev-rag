"""What the agents produce: Shortlists, Screening verdicts, Interview assessments and Compliance findings."""

from typing import Literal

from pydantic import BaseModel, Field


class ShortlistEntry(BaseModel):
    candidate_id: str
    name: str
    reason: str


class Shortlist(BaseModel):
    job_id: str
    candidates: list[ShortlistEntry] = Field(default_factory=list, max_length=8)


class CompetencyScore(BaseModel):
    competency: str
    score: int = Field(ge=1, le=5)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    rationale: str


class ScreeningVerdict(BaseModel):
    candidate_id: str
    job_id: str
    decision: Literal["advance", "reject"]
    scores: list[CompetencyScore] = Field(min_length=1)
    summary: str


class InterviewTurn(BaseModel):
    question: str
    answer: str


class InterviewAssessment(BaseModel):
    candidate_id: str
    job_id: str
    turns: list[InterviewTurn]
    scores: list[CompetencyScore]
    recommendation: Literal["hire", "no_hire", "next_round"]
    summary: str


class ComplianceFinding(BaseModel):
    check: Literal["input_guardrail", "consent", "groundedness", "policy"]
    subject: str
    passed: bool
    detail: str


class AgentRunNote(BaseModel):
    agent: str
    subject: str = ""
    stopped_reason: str
    steps: int
    cost_usd: float


UNSUPPORTED_KEYWORDS = {"title", "default", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "description"}


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
