# Benchmark judged by an independent Reference judge, not by Jev

The results will be published (LinkedIn + article), so the Level 2 vs Level 3 comparison must survive scrutiny. Quality is measured on a Golden set (synthetic questions plus a hand-checked subset) with retrieval metrics and a strong System 2 Reference judge for answers. Jev-as-a-Judge is not used to score the benchmark, because it would be grading the Level it belongs to; instead it is itself evaluated for agreement with the Reference judge. The Reference judge is always from a different model family than the System 2 model, to avoid self-preference bias.

## Considered Options

- LLM-as-judge only, no references: rejected, leaves retrieval quality unmeasured.
- Fully hand-written golden set: rejected for a portfolio budget; a hand-checked subset gives the credibility at lower cost.
