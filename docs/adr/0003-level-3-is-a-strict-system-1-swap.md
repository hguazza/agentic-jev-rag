# Level 3 is Level 2 with only the System 1 decisions swapped

Status: still applies, to the System 1 comparison: the LLM decider and the Jev decider are the only difference between runs (ADR 0009).

To make the Headline claim defensible, Level 3 reuses Level 2's retrieval (hybrid search, reranker), System 2 model and prompts unchanged; only the System 1 decisions move from the Level 2 baseline model to Jev. Any quality or latency difference is then attributable to the System 1 swap. Redesigns that exploit Jev differently (e.g. Jev chunk filtering replacing the reranker) are allowed only as separately labelled variants, never in the main comparison.
