# System 1 comparison

Run 2026-10-08T23:29:32, 4 concurrent requests per implementation. Latency is end-to-end through OpenRouter.

| decision | implementation | n | accuracy | p50 latency (s) | p95 latency (s) | mean cost ($) | errors |
|---|---|---|---|---|---|---|---|
| chunk_relevance | llm | 40 | 100% | 1.693 | 2.901 | 0.0003830 | 0 |
| chunk_relevance | jev | 40 | 98% | 0.452 | 0.587 | 0.0000174 | 0 |
| groundedness | llm | 30 | 97% | 1.603 | 1.988 | 0.0002933 | 0 |
| groundedness | jev | 30 | 93% | 0.456 | 0.524 | 0.0000148 | 0 |
| input_guardrail | llm | 60 | 97% | 1.654 | 2.153 | 0.0005192 | 0 |
| input_guardrail | jev | 60 | 98% | 0.450 | 1.013 | 0.0000231 | 0 |
| route | llm | 40 | 100% | 1.582 | 2.810 | 0.0003813 | 0 |
| route | jev | 40 | 100% | 0.470 | 0.846 | 0.0000187 | 0 |

| decision | Jev speed-up (p50) | Jev cost ratio (mean) | accuracy gap (Jev − LLM) |
|---|---|---|---|
| chunk_relevance | 3.7× | 22× cheaper | -3 pts |
| groundedness | 3.5× | 20× cheaper | -3 pts |
| input_guardrail | 3.7× | 22× cheaper | +2 pts |
| route | 3.4× | 20× cheaper | +0 pts |

Input guardrail errors that matter most:
- llm: 1 discriminatory requests allowed, 1 legitimate requests refused
- jev: 1 discriminatory requests allowed, 0 legitimate requests refused
