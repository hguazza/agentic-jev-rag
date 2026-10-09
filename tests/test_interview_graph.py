"""The live interview graph: briefing, Turn check routing, probes and Interview signals, with the fake gateway."""

from agentic_jev_rag.recruiter import interviewer
from agentic_jev_rag.recruiter.interview_graph import build_interview_graph
from tests.interview_scripts import interview_gateway, pending, reply, start


def count_documents(ctx):
    """Record every RAG document id the context fetches from now on."""
    fetched, original = [], ctx.rag.document

    def document(document_id):
        fetched.append(document_id)
        return original(document_id)

    ctx.rag.document = document
    return fetched


def test_the_briefing_is_fetched_once_per_interview(recruiter_context):
    ctx = recruiter_context(interview_gateway())
    graph = build_interview_graph(ctx)
    fetched = count_documents(ctx)

    start(graph, ctx)
    reply(graph, "I partitioned Kafka topics.")
    done = reply(graph, "I owned on-call.")

    assert pending(done) is None and done["assessment"] is not None
    assert [d for d in fetched if d.startswith("policy:")] == ["policy:interview-guide", "policy:scorecards"]


def test_the_briefing_has_the_jobs_competencies_and_the_dossier(recruiter_context):
    ctx = recruiter_context(interview_gateway())

    briefing = interviewer.brief(ctx, ctx.rag.document("C-001"), ctx.rag.document("J-001"))

    assert briefing.competencies == ("Technical depth", "System design", "Delivery", "Collaboration")
    assert "Interview Guide" in briefing.text and "Kafka" in briefing.text
    assert briefing.dossier.startswith("CV:\n") and "Kafka" in briefing.dossier


def test_an_occupation_without_a_scorecard_has_no_competencies():
    assert interviewer.competencies("## Sales\n- **Closing:** wins deals.", "astronaut") == ()
