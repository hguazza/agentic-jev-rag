"""Corpus build: public rows are sampled, normalised and enriched; synthetic notes link real ids."""

import json
from datetime import date
from pathlib import Path

from agentic_jev_rag.corpus.build import build_corpus
from agentic_jev_rag.corpus.public import clean_resume_text, job_occupation, read_jobs, read_resumes
from agentic_jev_rag.gateway import FakeModelGateway
from agentic_jev_rag.rag.chunking import chunk_document

PUBLIC = Path(__file__).parent / "fixtures" / "public"
TODAY = date(2026, 10, 8)


def enrichment_gateway():
    def complete(model, prompt, system, schema):
        name = schema[0] if schema else None
        if name == "candidate_profile":
            return json.dumps({"headline": "IT Specialist", "years_experience": 6, "skills": ["Python", "Networking", "SQL"]})
        if name == "job_posting":
            return json.dumps({
                "title": "IT Support Lead", "about": "We are a fictional client.", "responsibilities": "Run the help desk.",
                "requirements": "3+ years of IT support.", "benefits": "Pension.", "required_skills": ["Networking", "SQL"],
                "seniority": "mid",
            })
        return "Interview note: the candidate described relevant work. Scores: 4, 3, 4, 4. Outcome as recorded."

    return FakeModelGateway(complete_responder=complete)


def build(seed=7, gateway=None):
    return build_corpus(
        read_resumes(PUBLIC / "resumes.csv"), read_jobs(PUBLIC / "jobs.csv"), gateway or enrichment_gateway(),
        system_2_model="s2", extraction_model="s1", seed=seed, today=TODAY,
        resumes_per_occupation=2, jobs_per_occupation=2, notes_per_job=2,
    )


def test_resume_text_puts_section_headings_on_their_own_lines():
    text = clean_resume_text("         IT MANAGER       Summary    Ran IT.     Experience     Company Name    Did things.   Skills    Python")

    sections = [c.section for c in chunk_document(_doc(text), size=2000, overlap=100)]
    assert sections == ["body", "summary", "experience", "skills"]


def test_job_occupation_comes_from_the_posting_function():
    assert job_occupation({"function": "Accounting/Auditing"}) == "finance"
    assert job_occupation({"function": "Information Technology"}) == "information_technology"
    assert job_occupation({"function": "Marketing"}) is None


def test_fraudulent_and_out_of_scope_rows_are_filtered_out():
    jobs = read_jobs(PUBLIC / "jobs.csv")
    resumes = read_resumes(PUBLIC / "resumes.csv")

    assert all(row["occupation"] for row in jobs) and len(jobs) == 12  # 2 per function, 6 functions → 5 occupations
    assert {row["occupation"] for row in resumes} == {"information_technology", "human_resources", "sales", "design", "finance"}


def test_build_produces_candidates_jobs_and_notes_with_linked_ids():
    documents = build()
    by_collection = {c: [d for d in documents if d.collection == c] for c in ("candidates", "jobs", "interviews")}

    assert len(by_collection["candidates"]) == 10 and len(by_collection["jobs"]) == 10
    candidate = by_collection["candidates"][0]
    assert candidate.id == "C-0001"
    assert {"name", "occupation", "location", "years_experience", "skills", "consent_until", "source"} <= set(candidate.metadata)
    job = by_collection["jobs"][0]
    assert job.id == "J-001" and job.metadata["client"] and "Requirements" in job.text
    ids = {d.id for d in documents}
    for note in by_collection["interviews"]:
        assert note.metadata["candidate_id"] in ids and note.metadata["job_id"] in ids
        assert note.metadata["outcome"] in {"advanced", "rejected", "offered", "hired", "withdrew"}
        assert note.metadata["occupation_match"] is True


def test_candidate_locations_come_from_job_locations_so_location_filters_are_meaningful():
    documents = build()
    job_locations = {d.metadata["location"] for d in documents if d.collection == "jobs"}

    assert {d.metadata["location"] for d in documents if d.collection == "candidates"} <= job_locations


def test_build_is_deterministic_for_a_seed():
    assert build(seed=3) == build(seed=3)
    assert build(seed=3) != build(seed=4)


def _doc(text):
    from agentic_jev_rag.corpus.documents import Document

    return Document("C-1", "candidates", "T", text, {})
