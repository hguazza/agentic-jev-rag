Dear micro1 hiring team,

I am applying for the AI/ML Engineer, Internal Platforms role. To show how I would work on Zara, I built an autonomous AI recruiting agent of my own (GitHub). In it, a LangGraph Supervisor hands work to Sourcer, Screener, Interviewer and Compliance agents, which reach a hybrid-search RAG service only through HTTP tools. Both FastAPI services run on AWS ECS, and a LiveKit voice agent drives the Interviewer for live spoken interviews.

What I would bring to your recruiting agent:

- Multi-agent workflows and tool calling: at Deep, I build LangGraph agents on AWS Bedrock for a top-3 Brazilian bank, including TrainHub, a real-time voice agent whose tool calls into deterministic functions keep negotiation math correct.
- Evaluation as a release gate: eval benchmarks gate every deployment of our agents in CI/CD, and an LLM-as-a-judge checks a quality agent that replaces human call evaluation with 90%+ accuracy.
- Model choices backed by numbers: in my project, Jev matched Claude Haiku 4.5 within 3 points on four pipeline decisions at about 3.5x lower latency and 20x lower cost. At Deep, distilling Claude Sonnet into Qwen3-32B kept about 90% of its score at 40x lower cost.
- Assessments that hold up: fairness and consent rules are enforced in code, and every screening verdict passes a groundedness check against the CV evidence it cites.

The next eval I would add to my own system is conversational fidelity for the Interviewer: whether each follow-up responds to what the candidate actually said. Your posting names it directly, and it is the kind of problem I would like to solve with your product leads. I work remotely from São Paulo (UTC-3), which overlaps with most of the US working day.

I would welcome the chance to discuss how I can help ship the next version of Zara.

Kind regards,
Henrique Guazzelli
