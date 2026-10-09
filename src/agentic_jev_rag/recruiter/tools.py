"""RAG API endpoints exposed to agents as tools."""

from agentic_jev_rag.recruiter.agent_loop import Tool
from agentic_jev_rag.recruiter.rag_client import RagClient

COLLECTIONS = {"type": "array", "items": {"type": "string", "enum": ["candidates", "jobs", "interviews", "policies"]}}
FILTERS = {
    "type": "object",
    "properties": {
        "occupation": {"type": "string"},
        "location": {"type": "string"},
        "min_years_experience": {"type": "number"},
        "candidate_id": {"type": "string"},
        "job_id": {"type": "string"},
    },
}


def rag_tools(rag: RagClient) -> dict[str, Tool]:
    tools = [
        Tool(
            "search",
            "Hybrid search over SwiftAI's documents. Returns matching chunks with metadata.",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "collections": COLLECTIONS, "filters": FILTERS, "top_k": {"type": "integer"}},
                "required": ["query"],
            },
            rag.search,
        ),
        Tool(
            "match_candidates",
            "Rank Candidates (with valid consent) for a Job, each with the CV chunks that are their evidence.",
            {"type": "object", "properties": {"job_id": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["job_id"]},
            rag.match,
        ),
        Tool(
            "ask",
            "Ask a question and get a grounded answer with citations, e.g. about a policy or a candidate's history.",
            {
                "type": "object",
                "properties": {"question": {"type": "string"}, "collections": COLLECTIONS, "filters": FILTERS},
                "required": ["question"],
            },
            rag.ask,
        ),
        Tool(
            "get_document",
            "Fetch one document (a Candidate's CV, a Job, an interview note or a policy) by id.",
            {"type": "object", "properties": {"document_id": {"type": "string"}}, "required": ["document_id"]},
            rag.document,
        ),
    ]
    return {t.name: t for t in tools}
