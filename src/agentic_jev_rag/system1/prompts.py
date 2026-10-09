"""Instructions shared by both System 1 deciders, so the LLM and Jev answer the same questions (ADR 0003)."""

ROUTE_INSTRUCTIONS = "Which collection of SwiftAI Recruit documents should be searched to answer this question?"
ROUTE_CRITERIA = {
    "candidates": "CVs and candidate profiles: skills, experience, education, location of people in the talent pool.",
    "jobs": "Job descriptions from clients: responsibilities, requirements, seniority, location and benefits.",
    "interviews": "Interview notes and outcomes: what candidates said, scores, and whether they advanced, were rejected, offered or hired.",
    "policies": "SwiftAI rules and playbooks: fair hiring, data protection and consent, interview guide, scorecards, fast-track, AI oversight.",
}

GUARDRAIL_INSTRUCTIONS = (
    "Classify this request made to SwiftAI Recruit's AI recruiter. Judge what the request asks the recruiter "
    "to do, including any selection criteria it contains."
)
GUARDRAIL_CRITERIA = {
    "allowed": (
        "A legitimate recruiting request or question that uses only job-related criteria (skills, experience, "
        "qualifications, certifications, the job's location, availability, right to work) or asks about "
        "candidates, jobs, interviews or SwiftAI policies."
    ),
    "discriminatory": (
        "Asks to select, exclude, rank or score people by a protected characteristic (age, sex or gender, "
        "marital or family status, pregnancy or parental plans, race, ethnicity, colour, nationality, religion, "
        "disability or health, sexual orientation, union membership) or by a proxy for one, such as "
        "'digital natives', 'young and dynamic', 'recent graduates' for a non-graduate role, a maximum number "
        "of years of experience, 'no career gaps', 'native speaker', or graduation-year cut-offs."
    ),
    "out_of_scope": (
        "Not a recruiting task for SwiftAI: general knowledge, coding help, personal or legal advice, or "
        "anything unrelated to candidates, jobs, interviews or hiring policy."
    ),
}

RELEVANCE_INSTRUCTIONS = (
    "Is this passage useful evidence for answering the question? Answer yes only if the passage contains "
    "information that directly helps answer it."
)

GROUNDEDNESS_INSTRUCTIONS = (
    "Is every factual claim in the statement supported by the evidence passages? Answer no if any claim is "
    "missing from, or contradicted by, the evidence."
)
