# 14. An AI engineer showcase for micro1's role

Date: 2026-10-09. Status: accepted.

## Context

The project was built to show fit for micro1's AI/ML Engineer, Internal Platforms role (`docs/job-description.md`), but its corpus had no AI engineering Candidates or Jobs, and the public resume dataset has none.

## Decision

- Add J-061, micro1's role, as the one real Client in the corpus. Every other Client stays fictional.
- Append 12 synthetic Candidates (C-0201..C-0212) written from a Persona grid (3 strong, 4 with gaps, 3 adjacent, 2 bluffers) and the author (C-0213) with a real name, CV and Dossier. The rest of the corpus is unchanged.
- Personas never enter the corpus, because chunk payloads carry document metadata to the Screener. Every pool Candidate has `source: ai_pool`, so the metadata does not say which CV is the author's.
- The Recruiter ranks and screens all 13. It interviews the fakes the Screener advances through the Recruiter API, with a Simulated candidate playing each Persona. The author holds the same interview by voice.
- Same judge: every Interview assessment records the System 2 model, the System 1 decider and a hash of the Interview Guide and Scorecards. The showcase refuses an author interview graded under a different one.

## Consequences

- The comparison is honest about its limits: simulated text against live speech, Personas written by the same model family that grades, and one run per Candidate. The report states them.
- `corpus.build` drops the pool. Run `corpus.ai_pool` again after a rebuild.
