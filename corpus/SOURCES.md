# Corpus sources

- **candidates.jsonl**: résumé text from `opensporks/resumes` (Hugging Face, CC0-1.0), a mirror of the Kaggle
  LiveCareer "Resume Dataset" (CC0). Names, locations and Consent dates are synthetic. Headlines, years of
  experience and skills were extracted by the System 1 baseline.
- **jobs.jsonl**: postings from EMSCAD via `victor/real-or-fake-fake-jobposting-prediction` (Hugging Face, CC0-1.0),
  rewritten by the System 2 model for fictional Clients, with identifying details removed.
- **interviews.jsonl**: synthetic interview notes written by the System 2 model for seeded (Candidate, Job) pairs
  and Outcomes.
- **policies/**: hand-written SwiftAI Recruit policies.

Rebuild with `uv run python -m agentic_jev_rag.corpus.build --seed 7` (downloads are cached in `data/raw/`).
