# System 1 comparison

Run 2026-10-09T03:59:27, 4 concurrent requests per implementation. Latency is end-to-end through OpenRouter.

| decision | implementation | n | accuracy | p50 latency (s) | p95 latency (s) | mean cost ($) | errors |
|---|---|---|---|---|---|---|---|
| chunk_relevance | llm | 40 | 100% | 1.746 | 2.264 | 0.0003830 | 0 |
| chunk_relevance | jev | 40 | 98% | 0.584 | 0.665 | 0.0000174 | 0 |
| groundedness | llm | 30 | 100% | 1.779 | 2.190 | 0.0003421 | 0 |
| groundedness | jev | 30 | 97% | 0.568 | 0.667 | 0.0000167 | 0 |
| input_guardrail | llm | 60 | 97% | 1.839 | 2.322 | 0.0005192 | 0 |
| input_guardrail | jev | 60 | 98% | 0.580 | 0.688 | 0.0000231 | 0 |
| route | llm | 40 | 100% | 1.780 | 2.306 | 0.0003813 | 0 |
| route | jev | 40 | 100% | 0.586 | 0.649 | 0.0000187 | 0 |
| turn_check | llm | 60 | 98% | 1.815 | 2.225 | 0.0007266 | 0 |
| turn_check | jev | 60 | 95% | 0.570 | 0.634 | 0.0000311 | 0 |

| decision | Jev speed-up (p50) | Jev cost ratio (mean) | accuracy gap (Jev − LLM) |
|---|---|---|---|
| chunk_relevance | 3.0× | 22× cheaper | -3 pts |
| groundedness | 3.1× | 20× cheaper | -3 pts |
| input_guardrail | 3.2× | 22× cheaper | +2 pts |
| route | 3.0× | 20× cheaper | +0 pts |
| turn_check | 3.2× | 23× cheaper | -3 pts |

Input guardrail errors that matter most:
- llm: 1 discriminatory requests allowed, 1 legitimate requests refused
- jev: 1 discriminatory requests allowed, 0 legitimate requests refused

Turn check errors that matter most:
- llm: 0 prompt injections missed, 1 real answers reprompted or redirected
- jev: 0 prompt injections missed, 0 real answers reprompted or redirected
