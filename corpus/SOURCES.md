# Corpus sources

- **candidates.jsonl**: résumé text from `opensporks/resumes` (Hugging Face, CC0-1.0), a mirror of the Kaggle
  LiveCareer "Resume Dataset" (CC0). Names, locations and Consent dates are synthetic. Headlines, years of
  experience and skills were extracted by the System 1 baseline.
- **jobs.jsonl**: postings from EMSCAD via `victor/real-or-fake-fake-jobposting-prediction` (Hugging Face, CC0-1.0),
  rewritten by the System 2 model for fictional Clients, with identifying details removed.
- **interviews.jsonl**: synthetic interview notes written by the System 2 model for seeded (Candidate, Job) pairs
  and Outcomes.
- **policies/**: hand-written policies for the fictional agency.
- **AI engineer showcase pool** (J-061, C-0201..C-0213): J-061 is micro1's public job description, restructured
  into the Job sections. C-0201..C-0212 are synthetic CVs and Dossiers written from `ai_pool/personas.toml` (the committed
  pool was written in Claude Code; re-running the generator has the System 2 model write them). C-0213 is the author's
  own CV. A rebuild drops the pool: run `uv run python -m agentic_recruiter.corpus.ai_pool`.

Rebuild with `uv run python -m agentic_recruiter.corpus.build --seed 7` (downloads are cached in `data/raw/`).
