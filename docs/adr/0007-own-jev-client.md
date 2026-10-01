# Jev is called through our own thin client, not langchain-typesafe

`langchain-typesafe` (0.0.1a3) works against OpenRouter but discards the per-call `usage.cost`, and cost is part of the Headline claim. Jev is therefore called through a small client of our own against OpenRouter's `/api/v1/systemone`, behind the same System 1 interface as the System 1 baseline, so both are timed and costed by one code path. `langchain-typesafe` is mentioned in the article as the LangChain integration route.
