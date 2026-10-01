# Agentic Jev RAG

A public, re-runnable portfolio benchmark of three successive RAG architectures over EU legal text published in both English and Portuguese (English first), built to test one headline claim: Level 3 matches Level 2 answer quality at a fraction of the latency and cost.

## Language

### Architectures

**Level**:
One of the three RAG architectures under comparison: Level 1 (Vanilla RAG), Level 2 (Agentic RAG), Level 3 (Agentic Jev RAG).
_Avoid_: tier, version, mode

**System 1 decision**:
A fast, typed judgement inside the pipeline — routing, chunk relevance, risk classification, groundedness — that does not generate prose.
_Avoid_: classification step, fast path

The four System 1 decisions are:

- **Route**: which Collection(s) to search for a question.
- **Chunk relevance**: whether a reranked chunk is kept for generation.
- **Input guardrail**: whether a question is refused (out of scope, personal legal advice, contains personal data).
- **Groundedness check**: whether a final answer is supported by its cited chunks.

_Avoid_: "router" or "guardrail" for the whole System 1 layer.

**System 2 model**:
The generative LLM that writes answers and, in Level 2, also makes the System 1 decisions.
_Avoid_: main model, big model

**Model role**:
A named slot the pipeline needs a model for — System 2 model, System 1 baseline, Reference judge — filled from config rather than fixed in code.
_Avoid_: model type, model tier

**System 1 baseline**:
The small, fast generative model that makes System 1 decisions in Level 2, and that Jev replaces in Level 3.
_Avoid_: router model, cheap model

**Collection**:
One indexed corpus that a Route decision can choose between: Case law (CJEU judgments) or Legislation (EU acts).
_Avoid_: index, namespace, database

### Evaluation

**Language**:
The language of a corpus, its Golden set and its questions; each Language is benchmarked separately with the same pipeline.
_Avoid_: locale

**Benchmark run**:
One execution of all Levels over one Language's Golden set with one model configuration, producing quality, latency and cost figures.
_Avoid_: experiment, eval run

**Headline claim**:
"Level 3 matches Level 2 answer quality at a fraction of the latency and cost." The single statement the benchmark exists to confirm or refute.

**Golden set**:
The fixed set of Portuguese questions, each paired with the source document(s) that answer it; mostly synthetic, with a hand-checked subset.
_Avoid_: test set, dataset

**Hand-checked subset**:
The golden-set questions a human has verified, used as the credibility anchor when quoting results.

**Reference judge**:
A strong System 2 model that grades answers for the benchmark; never the system under test.
_Avoid_: evaluator

**Jev-as-a-Judge**:
Level 3's online evaluation feature, itself validated against the Reference judge rather than used to score the benchmark.
_Avoid_: judge (unqualified)
