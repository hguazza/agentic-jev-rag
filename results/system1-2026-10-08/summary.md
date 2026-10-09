# System 1 comparison

Run 2026-10-08T23:40:47, 4 concurrent requests per implementation. Latency is end-to-end through OpenRouter.

| decision | implementation | n | accuracy | p50 latency (s) | p95 latency (s) | mean cost ($) | errors |
|---|---|---|---|---|---|---|---|
| chunk_relevance | llm | 40 | 100% | 1.636 | 1.908 | 0.0003830 | 0 |
| chunk_relevance | jev | 40 | 98% | 0.452 | 0.503 | 0.0000174 | 0 |
| groundedness | llm | 30 | 100% | 1.589 | 1.852 | 0.0003423 | 0 |
| groundedness | jev | 30 | 97% | 0.422 | 0.499 | 0.0000167 | 0 |
| input_guardrail | llm | 60 | 95% | 1.601 | 2.041 | 0.0005196 | 0 |
| input_guardrail | jev | 60 | 98% | 0.438 | 0.547 | 0.0000231 | 0 |
| route | llm | 40 | 100% | 1.525 | 2.074 | 0.0003813 | 0 |
| route | jev | 40 | 100% | 0.440 | 0.601 | 0.0000187 | 0 |

| decision | Jev speed-up (p50) | Jev cost ratio (mean) | accuracy gap (Jev − LLM) |
|---|---|---|---|
| chunk_relevance | 3.6× | 22× cheaper | -3 pts |
| groundedness | 3.8× | 20× cheaper | -3 pts |
| input_guardrail | 3.7× | 22× cheaper | +3 pts |
| route | 3.5× | 20× cheaper | +0 pts |

Input guardrail errors that matter most:
- llm: 1 discriminatory requests allowed, 2 legitimate requests refused
- jev: 1 discriminatory requests allowed, 0 legitimate requests refused
