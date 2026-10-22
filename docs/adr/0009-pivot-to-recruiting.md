# Pivot from the EU legal benchmark to an AI recruiter

The project now builds an autonomous AI recruiter for a fictional recruiting agency, to show fit for an AI/ML engineering role on recruiting agents (`docs/job-description.md`). The legal benchmark is parked, not deleted (`docs/parked/eu-legal-rag-benchmark.md`). What carries over: the Model gateway over OpenRouter (ADR 0005), our own Jev client (ADR 0007), Docker Compose (ADR 0008), the System 1 decision vocabulary, and the rule that only the System 1 decider differs between the LLM and Jev runs (ADR 0003). What is dropped: Language as a dimension (ADR 0004), the EU corpus (ADR 0006) and the three-Level storyline. The LLM-versus-Jev comparison is kept as a System 1 comparison inside the Recruiter rather than as the project's headline.

## Considered Options

- Keep the three-Level benchmark and swap the corpus: rejected, because the target role asks for shipped agents and services more than a benchmark.
- Drop Jev entirely: rejected, because fast, cheap refusal of discriminatory requests is a strong recruiting story.
