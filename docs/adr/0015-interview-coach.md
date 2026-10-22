# 15. An Interview Coach for the showcase

Date: 2026-10-22. Status: accepted.

## Context

The author's voice interview for J-061 came back `next_round`, with answers that ran seven to nine minutes and an ECS deployment claim the Dossier contradicts. The assessment says what went wrong for a recruiter; it does not tell the candidate what to do differently.

## Decision

- A Coach (`agentic_recruiter.coach`) debriefs one finished voice interview for the candidate: per question, what it tested, what worked, what cost points and a tighter model answer; then up to three delivery habits and what to say instead of each red flag.
- Showcase only. It runs on demand on a saved results file; it is not part of the interview graph and nothing it writes reaches a recruiter or a score.
- Model answers use only facts from the candidate's CV and Dossier. The configured System 1 decider's groundedness check, the one the Compliance agent uses, checks each one; an ungrounded answer is rewritten once and, if still ungrounded, kept and marked.
- Timing coaches delivery only. ADR 0013 still holds: the assessment never used it, and the Coach says so.

## Consequences

- Candidate-facing coaching on the platform stays open: feedback would have to wait for the human review of the assessment, so it does not pre-empt a decision a recruiter has not made.
- Coaching quality depends on the System 2 model, and the groundedness check covers the model answers, not the critique.
