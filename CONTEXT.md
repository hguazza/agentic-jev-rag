# Agentic Jev RAG: SwiftAI Recruiter

An autonomous AI recruiter for SwiftAI Recruit (`docs/company-overview.md`): a production RAG service over recruiting documents, and a LangGraph multi-agent Recruiter that calls it as a tool. The Recruiter's fast typed checks, its System 1 decisions, can be made by a small LLM or by Jev, and the project measures how the two compare on accuracy, latency and cost. Built to demonstrate the role in `docs/job-description.md`. The earlier EU legal benchmark is parked in `docs/parked/eu-legal-rag-benchmark.md`.

## Language

### Corpus

**Collection**:
One indexed group of documents that retrieval can search: Candidates, Jobs, Interviews or Policies.
_Avoid_: index, namespace, database

**Candidate**:
A person in the talent pool. Their Profile is their CV plus metadata (occupation, years of experience, location, skills, consent date).
_Avoid_: applicant, user, resume (for the person)

**Job**:
An open or past role at a Client, with its description, required skills, seniority and location.
_Avoid_: vacancy, position, req

**Client**:
A company that hires through SwiftAI Recruit. Clients are fictional.

**Interview note**:
A record of one interview stage for a Candidate and a Job: what was asked, what was observed, the scores and the Outcome. Interview notes are synthetic.

**Outcome**:
What happened to a Candidate for a Job: advanced, rejected, offered, hired or withdrew.

**Policy**:
A SwiftAI rule or playbook the Recruiter must follow: fair hiring, data protection and consent, interview guides, Scorecards, fast-track rules.

**Scorecard**:
The competencies a Job is assessed on, each rated on a fixed scale.

**Consent**:
A Candidate's permission for SwiftAI to hold and use their Profile, valid until a recorded date. A Candidate without valid Consent must not be contacted or shortlisted.

### RAG service

**Search**:
Hybrid retrieval (dense plus sparse) over one or more Collections, with metadata filters and optional reranking. It returns chunks with scores.

**Match**:
Ranking Candidates for a Job, each with the chunks that are its Evidence.

**Ask**:
A grounded answer to a question, with citations, made by the System 2 model after the System 1 decisions.

**Evidence**:
The cited chunks behind a ranking, an answer or a verdict.

### Agents

**Recruiter**:
The whole LangGraph multi-agent system that takes a hiring goal and returns a Recruitment report.
_Avoid_: bot, assistant

**Supervisor**:
The agent that reads the goal and hands work to the other agents until the goal is done.

**Sourcer**:
The agent that finds and ranks Candidates for a Job using Match and Search.

**Screener**:
The agent that assesses one Candidate against a Job's Scorecard and gives a Screening verdict.

**Screening verdict**:
Advance or reject, with a score per Scorecard competency and the Evidence for each.

**Interviewer**:
The agent that holds a text interview with a Candidate, asking questions from the Job's competencies and the Candidate's history, and gives an Interview assessment.

**Simulated candidate**:
An LLM playing a Candidate from their Profile, so interviews can run without a human. Used in demos and evaluations only.

**Compliance**:
The agent that checks the goal, the shortlist and the verdicts against Policy: fair hiring, Consent and grounded verdicts.
_Avoid_: guardrail (for the whole agent)

**Recruitment report**:
The Recruiter's final output for a goal: the shortlist, Screening verdicts, Interview assessments, Compliance findings and the System 1 trace.

### System 1

**System 1 decision**:
A fast, typed judgement inside the pipeline that does not generate prose.
_Avoid_: classification step, fast path

The four System 1 decisions are:

- **Route**: which Collection(s) to search for a question.
- **Chunk relevance**: whether a retrieved chunk is kept as Evidence.
- **Input guardrail**: whether a request is refused: discriminatory (selects or excludes people by a protected characteristic) or out of scope.
- **Groundedness check**: whether an answer or verdict is supported by its Evidence.

**System 1 decider**:
The component that makes the four System 1 decisions. It has two implementations: the LLM decider (the System 1 baseline) and the Jev decider. Everything else is the same whichever one runs.

**System 2 model**:
The generative LLM that writes answers and drives the agents.
_Avoid_: main model, big model

**System 1 baseline**:
The small, fast generative model that the LLM decider prompts with structured output.
_Avoid_: router model, cheap model

**Model role**:
A named slot the pipeline needs a model for (System 2 model, System 1 baseline, Reference judge, embedding model, reranker, Jev model), filled from config rather than fixed in code.
_Avoid_: model type, model tier

**System 1 trace**:
The list of System 1 decisions made for one request: decision, outcome, latency, cost and implementation.

### Evaluation

**Labelled set**:
A committed file of examples with a known right answer for one System 1 decision, used to score both System 1 deciders.
_Avoid_: test set, dataset

**System 1 comparison**:
One run of both System 1 deciders over the Labelled sets, producing accuracy, latency and cost per decision.
_Avoid_: experiment, benchmark run

**Reference judge**:
A strong model from a different family than the System 2 model, used to label or grade; never the system under test.
_Avoid_: evaluator
