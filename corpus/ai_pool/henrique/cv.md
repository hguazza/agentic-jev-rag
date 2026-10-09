AI ENGINEER
São Paulo, SP, Brazil (remote) · github.com/hguazza

Summary
AI Engineer shipping production LLM agents in Python, FastAPI and LangGraph on AWS. Recently built an autonomous AI recruiter end to end: a five-agent LangGraph system (sourcing, screening, live interviewing, compliance) over a hybrid-search RAG service, deployed on AWS ECS, with a live voice interview demo on LiveKit. At work, I build multi-agent workflows with tool calling for a top-3 Brazilian bank on AWS Bedrock, and gate every deployment on eval benchmarks. I choose models by measuring accuracy, latency and cost, from distilling Claude Sonnet into Qwen3-32B (~90% of the score at 40x lower cost) to benchmarking a specialised decision model against a small LLM (~3.5x faster, ~20x cheaper).

Core Competencies
- Multi-Agent Workflows & Tool Calling: LangGraph supervisor and fan-out/fan-in graphs, LangChain, LlamaIndex, tool calling with deterministic guardrails, prompt chains, structured outputs (Pydantic), MCP; OpenAI, Anthropic, Bedrock, OpenRouter.
- RAG Architectures & Vector Databases: hybrid search (dense + sparse, RRF), reranking, grounded answers with citations; Qdrant, Elasticsearch, pgvector; evaluated with RAGAS.
- Evaluation Pipelines: labelled sets, deterministic model evaluation, LLM-as-a-judge, groundedness checks, retrieval metrics (recall@k, MRR), eval benchmarks gating CI/CD, Langfuse tracing.
- Python & FastAPI Microservices: low-latency async APIs (asyncio, bounded concurrency), OpenAPI, PostgreSQL, Redis, MongoDB, event-driven pipelines (SQS, RabbitMQ), Docker, API gateways (APISIX, LiteLLM).
- AWS: Bedrock, ECS, EC2, Lambda, S3, SQS in production; Terraform; AWS Certified Cloud Practitioner. Also Azure and GCP.

Experience
01/2026 to present
AI Engineer
Deep - Brazil
Voice AI platform for call-center debt collection, delivered for a top-3 Brazilian bank.
- Built a fraud detection agent system on Elasticsearch with asyncio and semaphore-bounded concurrency: 5,000+ calls analyzed in 100 minutes, R$5M+ in fraud detected, replacing manual analysis.
- Built a monitoring and quality agent that replaces human call evaluation with 90%+ accuracy, a multi-agent system that accelerates evaluations, and an LLM-as-a-judge that catches hallucinations in the agent's explanations; set up Bitbucket Pipelines CI/CD where eval benchmarks gate every deployment.
- Built TrainHub, a real-time voice agent that role-plays clients to score and train new operators (LiveKit, Deepgram, ElevenLabs), with tool calling into deterministic functions as guardrails for negotiation math.
- Run the agents on AWS: LangGraph agents call models through Bedrock, a self-deployed LiteLLM gateway runs on EC2, service endpoints run on ECS, and S3 to SQS to Lambda pipelines handle ingestion.
- Built LangGraph state graphs with parallel agents sharing state (fan-out/fan-in); built RAG on Qdrant with hybrid search and reranking, evaluated with RAGAS; distilled Claude Sonnet outputs into a fine-tuned Qwen3-32B reaching ~90% of Sonnet's score at 40x lower cost.
- Optimized the audio ML pipeline (300,000 call audios per month on RabbitMQ and SQS), migrating from NiFi to Airflow and fine-tuning Whisper from 27% to 12% WER; implemented guardrails (prompt injection, PII masking, LGPD, audit logging).
- Self-deployed an Apache APISIX gateway unifying 12 platforms' FastAPI services; introduced LangGraph and Langfuse to the AI team; mentor engineers and review code.

2023 to 2025
Independent Professional, AI, Automation & Data Engineering
Freelance - Brazil
- Built an LLM chatbot with retrieval-augmented generation (RAG), and AI agents and automations for commercial and operational tasks.
- Delivered an end-to-end analytics engineering project, from multi-source data extraction to a Looker dashboard.

01/2023 to 05/2023
Data Science Intern
Facio - Brazil
- Reduced default rates by 5%, saving over R$10,000 per month; built a One Big Table of all transactional data and a referral-campaign prediction model.

06/2021 to 02/2022
Commercial Analyst
Nutri Indústria e Comércio de Ração Ltda. - Brazil
- Automated routine tasks (40 minutes saved per day) and built sales dashboards; new products and clients drove a ~15% sales increase.

Featured Project
10/2026
Autonomous AI Recruiting Agent (independent project, GitHub)
- Built a Recruiter API (FastAPI and LangGraph) where a Supervisor hands work to Sourcer, Screener, Interviewer and Compliance agents. The agents reach a separate RAG API only through HTTP tools and return a cited recruitment report. A full run takes about 3 minutes and costs $0.38.
- Built the RAG API (Search, Match, Ask) on Qdrant: hybrid retrieval with dense and own sparse vectors fused with RRF, a reranker, and answers with citations, over 200 CVs, 60 jobs, 120 interview notes and hiring policies.
- Enforced fairness and safety in code, not prompts: a guardrail refuses discriminatory goals before any agent runs, candidates without consent are excluded, agents finish through Pydantic-validated submit tools, and every screening verdict passes a groundedness check.
- Benchmarked the pipeline's four fast decisions (routing, chunk relevance, input guardrail, groundedness) over 170 hand-labelled examples: Jev matched Claude Haiku 4.5 within 3 points while running ~3.5x faster (p50 0.44s vs 1.6s) and ~20x cheaper per decision.
- Deployed both APIs on AWS ECS; a LiveKit voice agent drives the Interviewer API for live spoken interviews. 104 tests at the public seams, with the model gateway (cost, latency and retries per call) as the only fake.

Education
2026 - Agentic AI Program, Johns Hopkins University (online)
2025 to 2026 - Postgraduate Degree in Machine Learning Engineering, FIAP, Brazil. APIs with FastAPI and Flask; ML model deployment; batch and streaming ETL on AWS with Terraform.
2021 to 2024 - Technologist Degree in Big Data & Analytical Intelligence, PUCPR, Brazil. Kafka and Spark streaming; Hadoop MapReduce in Java; ML on AWS EMR with Spark MLlib.
2015 to 2018 - Computer Engineering (not completed), Insper, São Paulo, Brazil

Certifications & Languages
- AWS Certified Cloud Practitioner; Cambridge Advanced English (CAE); Sales Management, Fundação Getúlio Vargas (FGV).
- Portuguese (native), English (advanced, C1).
