# Interview Coach: design

Date: 2026-10-22. Vocabulary: `CONTEXT.md`. Builds on `2026-10-09-voice-interview-design.md` and `2026-10-09-ai-engineer-showcase-design.md`.

## Goal

After the author's voice interview for J-061 (C-0213), a Coach writes a debrief the author can act on: for each question, what it was testing, what the answer did well, what cost points, and a tighter model answer built only from facts in the author's CV and Dossier; then up to three delivery habits to fix. The debrief is committed with the showcase.

Success: `results/ai-engineer-showcase/coaching.md` tells the author exactly what to change in the next interview, with every point quoting a specific turn, and every model answer passes the System 1 groundedness check against the CV and Dossier or is marked as not fully supported.

## Decisions

- **Showcase only.** The Coach runs on demand on one saved interview. It is not a node in the interview graph, it never runs on simulated interviews, and nothing it writes reaches a recruiter or a score.
- **Model answers use the candidate's own facts.** A model answer may rephrase, shorten and restructure, but every claim must come from the CV or Dossier. A red flag becomes advice on what to say instead (for example: say the APIs run in Docker Compose, or deploy them to ECS first), never a repeat of the unsupported claim.
- **Grounded by System 1.** Each model answer is checked with the configured System 1 decider's `groundedness`, the decision the Compliance agent uses for Screening verdicts.
- **Timing informs delivery, never scores.** Answer duration and words per minute feed the delivery habits only. ADR 0013 still holds: the assessment ignored them, and the Coach says so.
- **Out of scope**: candidate-facing coaching on the platform (it would have to wait for the human review of the assessment), a spoken debrief, coaching the simulated Candidates, comparing the author with other Candidates (that is `report.md`'s job).

## Input

A finished voice interview results file (`results/voice-interviews/<id>.json`, written by the voice worker): `assessment.turns` (each with `question`, `answer`, `competency`, `kind`, `scored`, `check`, `timing`), `assessment.scores`, `recommendation`, `summary`, `red_flags` and `integrity_signals`. The Candidate document (CV text and `metadata.dossier`) comes from `corpus/candidates.jsonl`, rendered with `interviewer.dossier_text`. The competencies come from `corpus/policies/scorecards.md` with `interviewer.competencies` and the Candidate's occupation.

## Module: `src/agentic_jev_rag/coach.py`

`uv run python -m agentic_jev_rag.coach results/voice-interviews/<id>.json [--out results/ai-engineer-showcase]`

It uses the configured `system_2_model` and System 1 decider (`make_deciders(config, gateway)[config.system_1]`). No new Model role, and no RAG API or interview graph.

1. **Group turns.** A reprompt or redirect is folded into the question it repeats, so one question with its retries gives one debrief over all its answers. Turns flagged as injection are left out.
2. **Debrief each question**, one System 2 call with structured output:
   - `looking_for`: what the question was testing, in terms of its competency;
   - `what_worked` and `what_cost_points`: lists of points, each with a quote from the answer;
   - `model_answer`: about 60 to 90 seconds of speech, first person;
   - `facts_used`: the CV and Dossier lines the model answer relies on.

   The prompt carries the question, the answers, the competency, its score and rationale, the red flags quoting that turn, the CV and the Dossier.
3. **Check groundedness**: `decider.groundedness(model_answer, facts_used + [cv, dossier])`. If ungrounded, one rewrite of that debrief with the verdict in the prompt, then a second check. If still ungrounded, the debrief is kept and marked `grounded: false`. Every decision goes into the trace.
4. **Delivery habits**, one System 2 call over the whole interview: up to three habits, each with the turn numbers that show it and a concrete fix. The prompt carries each turn's timing, the reprompts, the red flags and the ADR 0013 note that timing never affected the score. Each red flag gets a "what to say instead".

## Output (in `--out`)

- `coaching.md`:
  - the headline: recommendation, scores per competency, and the assessment summary;
  - the delivery habits, then the red flags with what to say instead;
  - one section per question: the question, what it was looking for, the answer quoted, what worked, what cost points, the model answer (marked "not fully supported by your CV" when `grounded` is false);
  - at the end: the System 1 decider, the cost of the run and the interview id.
- `coaching.json`: the same content, with the System 1 trace.
- `showcase.render_report` adds one link to `coaching.md` when the file exists in the same directory.

## Error handling

- **Unfinished interview** (`finished` false, `assessment` null, or `error` set): stop before any model call, with a message that says which.
- **Candidate missing from the corpus, or without a Dossier**: stop before any model call, naming the id.
- **No question with an answer to coach**: write a short `coaching.md` that says so, with no model call.
- **System 2 call fails** after the gateway's retries: stop, write nothing, name the question that failed.
- **Malformed structured output**: one retry, as `interviewer._structured` does, then stop.
- **Ungrounded twice**: a result, not an error (see step 3).

## Testing

`tests/test_coach.py`, with `FakeModelGateway` and scripted replies:

- **Debriefs**: one call per question; the prompt has the question, the answers, the competency, the CV and the Dossier; reprompted turns fold into one debrief; injection turns are left out.
- **Groundedness**: grounded first time, no rewrite and one trace entry; ungrounded once, exactly one rewrite whose prompt carries the verdict; ungrounded twice, `grounded: false` and the mark in `coaching.md`.
- **Delivery**: the prompt has each turn's timing, the reprompts, the red flags and the ADR 0013 note; every red flag has a "what to say instead".
- **Refusals**: an unfinished interview, an unknown Candidate and a Candidate without a Dossier each stop before any model call.
- **Rendering**: `coaching.md` has the headline, the habits and one section per question with quotes; `report.md` links to it only when it exists.
- **Live** (`-m live`): one real run on `tests/fixtures/voice/c0213-j061.json`, a trimmed copy of the author's interview.

## Docs

- ADR 0015: the Interview Coach: showcase only, model answers from the candidate's own facts and checked by System 1, timing for delivery and never for scores, candidate-facing coaching left open until after human review.
- `CONTEXT.md`: **Coach**, the agent that debriefs a candidate on a finished interview; showcase only, never seen by a recruiter, never part of a score.
- README: in the "How I compare" section the showcase adds, link `coaching.md` and give the command to regenerate it.
