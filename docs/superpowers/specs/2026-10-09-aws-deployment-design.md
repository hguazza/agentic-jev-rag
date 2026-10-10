# AWS deployment: design

Date: 2026-10-09. Vocabulary: `CONTEXT.md`. Decision record: ADR 0014. Supersedes "AWS deployment is out of scope" in `2026-10-08-recruiter-design.md` and the "persistent checkpoints" exclusion in `2026-10-09-voice-interview-design.md`.

## Goal

Run the Recruiter on AWS the way a production team would: microservices on ECS Fargate, the long recruitment run on the runtime built for agents, everything in Terraform, deployed by CI without stored AWS keys. A reviewer can open one HTTPS URL, authenticate with an API key, run a recruitment, and hold a live voice interview from the LiveKit Agents Playground against the deployed stack.

It ships in two phases. Phase 1 puts everything on ECS and is complete on its own. Phase 2 is optional: it moves only the recruitment run to AgentCore, behind an unchanged API.

## Decisions (made by the agent, overridable)

- **Region `us-east-1`**, one environment (`prod`), Graviton (arm64) everywhere. Cheaper than `sa-east-1`, close to OpenRouter and Qdrant Cloud, and the images build natively on Apple silicon.
- **Each service on the runtime that fits it** (ADR 0014): the RAG API and the Voice worker on ECS Fargate; the Recruiter API on ECS as the public front door, with the interview graph in-process because every spoken turn waits on it; the recruitment run graph moves to Bedrock AgentCore Runtime in phase 2.
- **The Recruiter API keeps its contract.** Clients, the Voice worker and the OpenAPI docs do not change when the run moves. The one deliberate change is `/runs` becoming asynchronous (below).
- **Durable checkpoints in DynamoDB** (`DynamoDBSaver` from `langgraph-checkpoint-aws`), in both phases. Interviews survive deploys, task loss and scaling, and the Recruiter API can run more than one task. Locally and in tests the checkpointer stays in memory.
- **Qdrant Cloud** (free cluster, AWS `us-east-1`) instead of self-hosting Qdrant: a stateful database is not what this project showcases. The client gains an API key.
- **OpenRouter stays the Model gateway** (ADR 0005). Bedrock models are out of scope; the gateway seam makes them a later addition.
- **HTTPS through CloudFront's default domain** (`*.cloudfront.net`), with the ALB internal and reached through a CloudFront VPC origin. A custom domain is a later swap to Route 53 + ACM.
- **Out of scope**: multiple environments, a custom domain, Bedrock models, LLM tracing (LangSmith/Langfuse), multi-region.

## Architecture

```
Reviewer ─HTTPS─▶ CloudFront (+ WAF rate limit) ─VPC origin─▶ internal ALB
                                                                 │
                         ┌───────────────────────────────────────┘
                         ▼
  ┌──────────────── VPC, us-east-1, 2 AZs ─────────────────────────────────────┐
  │ private subnets                                                            │
  │   recruiter-api (ECS, 2 tasks) ── interview graph: in-process, always      │
  │        │                         run graph, phase 2: async ─▶ AgentCore    │
  │        │                                                 Runtime (VPC mode)│
  │        ├─ checkpoints ─▶ DynamoDB ◀─ checkpoints ───────────────┘          │
  │        └─ rag-api.swiftai.internal ─▶ rag-api (ECS, 2 tasks) ─▶ Qdrant Cloud│
  │   voice-worker (ECS, 1 task) ─▶ recruiter-api.swiftai.internal             │
  │        └─ outbound WSS ─▶ LiveKit Cloud ◀─ Agents Playground               │
  │ public subnets: NAT gateway                                                │
  └────────────────────────────────────────────────────────────────────────────┘
  Secrets Manager · ECR · CloudWatch (logs, Container Insights, alarms) · Budgets
```

## Network

- A dedicated VPC over two AZs: public subnets hold only the NAT gateway; tasks, the internal ALB and AgentCore's network interfaces sit in private subnets.
- **One NAT gateway**, not one per AZ: it is a single point of failure for egress, and that is a deliberate cost trade documented here. Every service needs internet egress anyway (OpenRouter, Qdrant Cloud, LiveKit).
- An S3 gateway endpoint (free) carries ECR image layers. Interface endpoints (ECR API, Logs, Secrets Manager) are skipped: about $45/month for traffic the NAT already carries.
- **Service discovery through Cloud Map DNS** (`swiftai.internal`), not Service Connect: Cloud Map names resolve from anything in the VPC, including AgentCore's interfaces in phase 2, so one mechanism serves both phases and no internal ALB is needed for the RAG API.
- Security groups: the ALB is internal and accepts port 80 from the VPC's own range only (CloudFront's VPC origin reaches it through interfaces inside the VPC); recruiter-api accepts the ALB and the Voice worker; rag-api accepts recruiter-api (and the AgentCore security group in phase 2).

## Services (phase 1)

| Service | Image | Size (vCPU / GB) | Tasks | Exposure |
|---|---|---|---|---|
| recruiter-api | `api` | 0.5 / 1 | 2, autoscaled on request count | internal ALB ← CloudFront |
| rag-api | `api` | 0.5 / 1 | 2, autoscaled on CPU | `rag-api.swiftai.internal:8000` |
| voice-worker | `voice` | 1 / 2 | 1 | none (outbound only) |
| ingest | `api` | 0.5 / 1 | one-off `RunTask` | none |

- **Two images from one Dockerfile** via build targets: `api` (today's image) and `voice` (adds the `voice` dependency group and bakes the Silero and turn-detector model files with `download-files`, so tasks don't download at start).
- ECR repositories with immutable tags (the git SHA) and scan on push.
- Deployment circuit breaker with automatic rollback on every service. Health checks hit `/health`.
- The Voice worker gets a 120 s stop timeout so a deploy can let a short interview finish; because checkpoints are durable, an interview cut off by a deploy can be resumed by the Candidate replying again.
- **Ingest is a one-off ECS task**, run by CI when `corpus/` changes and by hand with `infra/scripts/ingest.sh`.

## The Recruiter API on AWS

- **API key authentication.** Every route but `/health` requires `X-API-Key`, checked by FastAPI against a key from Secrets Manager. It is declared as an OpenAPI security scheme so Swagger UI's Authorize button works. Unset locally, auth is off. The Voice worker sends the same key.
- **`/runs` becomes asynchronous.** A full run takes about 3 minutes, past CloudFront's 60 s origin timeout. `POST /runs` returns `202` with a `run_id`; `GET /runs/{run_id}` returns `running`, `complete` (with the Recruitment report) or `failed`. Run records (status, report, error) live in a DynamoDB `runs` table, so any task can answer the status call. In phase 1 the run executes on a background thread of the task that accepted it; a task stopped mid-run marks it `failed` after a staleness window. Phase 2 removes this limit.
- **Interviews unchanged**: `POST /interviews` and `POST /interviews/{id}/reply` stay synchronous (one System 2 call per reply, well under the timeout) and now read and write DynamoDB checkpoints.
- Config gains `persistence = "memory" | "dynamodb"` (checkpoints and run records together), `CHECKPOINT_TABLE`, `RUNS_TABLE`, `DYNAMODB_ENDPOINT` (DynamoDB Local), `RECRUITER_API_KEY` and `QDRANT_API_KEY`, through the same environment overrides as `QDRANT_URL`.

## Phase 2 (optional): the recruitment run on AgentCore Runtime

Only `/runs` moves. Interviews stay in-process on ECS: a voice turn is latency-bound, and AgentCore would add a hop to every turn and a microVM cold start on the first turn and after 15 idle minutes, which is the wait Jev on the hot path exists to remove. With DynamoDB checkpoints, the in-process interview graph already survives deploys and scales across tasks, so AgentCore would add cost and latency without adding resilience.

The run is the opposite shape: about 3 minutes, five agents, mostly waiting on models, started and polled. That is what AgentCore is for.

- **An `agent` image** (arm64) serves AgentCore's contract (`POST /invocations`, `GET /ping` on port 8080) with one operation, `run`. It builds the run graph with the same DynamoDB checkpointer.
- **A seam in the Recruiter API**: `RunExecutor` with two adapters, `LocalRuns` (a background thread; local, tests, phase 1) and `AgentCoreRuns` (`InvokeAgentRuntime` through boto3). Config `runs = "local" | "agentcore"` picks one. `POST /runs` and `GET /runs/{run_id}` call the seam and nothing else changes.
- **Asynchronous invocation** (up to 8 hours), independent of API deploys, which removes phase 1's background-thread limit. The session id is `run-<run_id>` (AgentCore session ids are at least 33 characters).
- **`GET /runs/{run_id}` reads the run record**, which the agent writes when it finishes, so the API never waits on or calls the agent to report status.
- **VPC mode** in the private subnets, so the agent reaches `rag-api.swiftai.internal` without exposing the RAG API.
- **Inbound auth is IAM (SigV4)**: only the recruiter-api task role may invoke the runtime. No JWT authorizer is needed because no external client calls it.
- **Observability**: AgentCore's OpenTelemetry traces to CloudWatch GenAI observability, one trace per run showing each agent's steps.
- **Why DynamoDB and not AgentCore Memory for checkpoints**: one checkpointer serves both runtimes and both graphs, and the table is inspectable. AgentCore Memory is the right place for long-term memory across interviews, which this design doesn't need yet.

If time runs short, phase 1 ships alone and this section stays as the recorded next step.

## Secrets and configuration

- Secrets Manager holds `OPEN_ROUTER_API_TOKEN`, `QDRANT_API_KEY`, `RECRUITER_API_KEY` and the three `LIVEKIT_*` values. Terraform creates the secrets; their values are set out of band (`infra/scripts/put-secrets.sh` reads a git-ignored `.env.aws`), so no secret is in Terraform state or in git.
- ECS injects them through the task definitions' `secrets`; AgentCore receives non-secret config as environment variables and reads secrets at start through its IAM role.

## Observability and cost guardrails

- CloudWatch Logs (14-day retention) and Container Insights. ALB and CloudFront access logs are left out until something needs them.
- Alarms, to an SNS email topic: ALB 5xx rate, recruiter-api p95 latency, ECS running tasks below desired, DynamoDB throttles, and (phase 2) AgentCore invocation errors.
- WAF on CloudFront: a per-IP rate limit and AWS managed common rules. With the API key, this is the defence against someone spending the OpenRouter credit.
- An AWS Budget at $60/month with alerts at 50/80/100%.

## Terraform layout (`infra/`)

```
infra/
  bootstrap/        state bucket (S3, versioned, encrypted, native lockfile) and the GitHub OIDC deploy role; applied once by hand
  main/             network.tf, ecr.tf, ecs.tf, services.tf, edge.tf (CloudFront, WAF, ALB), data.tf (DynamoDB),
                    secrets.tf, observability.tf, iam.tf, agentcore.tf (phase 2), variables.tf, outputs.tf
  scripts/          push-images.sh, put-secrets.sh, ingest.sh
```

- Terraform ≥ 1.10 (S3-native state locking) and AWS provider 6.x (≥ 6.68 for AgentCore resources). The local Terraform is 1.5.7 and needs upgrading.
- `var.running` (default `true`) scales every service to zero and removes the NAT gateway, for between demos; CloudFront, the ALB and DynamoDB stay.
- No Terraform modules from the registry beyond what removes real boilerplate; resources stay readable in a handful of files.

## CI/CD (GitHub Actions)

- **On pull requests and pushes**: `uv run pytest` (Qdrant and DynamoDB Local as service containers), `terraform fmt -check`, `terraform validate`, `tflint` and `trivy config`. No AWS access: a `terraform plan` on pull requests would need a role that pull requests can assume, which is left out.
- **After CI passes on `main`**: build the `api`, `voice` (and phase 2 `agent`) images on arm64 runners, push to ECR tagged with the SHA, `terraform apply -var image_tag=<sha>`, run ingest if `corpus/` changed, then smoke-test `GET /health` through CloudFront.
- AWS access through **GitHub OIDC** into a role that only this repository's `production` environment (restricted to `main`) can assume; no AWS keys in GitHub. The role has `AdministratorAccess` because Terraform creates IAM roles; scoping it down is a known follow-up.

## Cost (us-east-1, running 24/7, rough)

| Item | Phase 1 / month |
|---|---|
| Fargate (2 + 2 tasks at 0.5/1, 1 at 1/2, arm64) | ~$86 |
| NAT gateway (1) | ~$33 + data |
| Internal ALB | ~$17 |
| WAF | ~$7 |
| CloudWatch, Secrets Manager, DynamoDB on-demand, ECR | ~$10 |
| **Total** | **~$155** |

With `running = false`, about $25. Phase 2 adds AgentCore's per-second charge for runs only: active CPU (near zero while agents wait on the LLM) plus memory while the session lives.

## Testing

- The Model gateway stays the only fake (AGENTS.md). DynamoDB Local joins Qdrant in `docker-compose.yml`, and the checkpointer tests run against it.
- New tests: the DynamoDB checkpointer resumes an interview across two app instances; `/runs` returns 202 and its status moves to `complete`; the API key is required on every route but `/health`; the `agent` container's `run` invocation (phase 2, in-process).
- `-m live` adds: `/health`, an authenticated interview and a run against the deployed CloudFront URL; `AgentCoreRuns` against the deployed runtime (phase 2).

## Docs

- ADR 0014 records the runtime split; ADR 0008 gains a line saying Compose remains the local runtime and AWS is the deployed one.
- README: a "Deploy to AWS" section (bootstrap, secrets, first apply, ingest, voice demo against AWS) and the architecture diagram.
- `CONTEXT.md`: add **Agent runtime** (where a Recruiter graph executes: in-process on ECS, or AgentCore for runs).

## Build order (each step deployable)

1. Code: Qdrant API key, DynamoDB checkpointer, API key auth, async `/runs`, `voice` build target. Tests green locally.
2. `infra/bootstrap`, then `infra/main` network, ECR, ECS, Cloud Map, DynamoDB, secrets; rag-api and recruiter-api reachable inside the VPC; ingest task.
3. Edge: internal ALB, CloudFront VPC origin, WAF. The public URL works with an API key.
4. voice-worker service; an interview from the Agents Playground against AWS.
5. Observability, alarms, budget; GitHub Actions with OIDC.
6. Phase 2 (optional): `agent` image, `RunExecutor` seam, AgentCore runtime in VPC mode, switch `runs = "agentcore"`.

## Prerequisites

- A working AWS CLI profile for the target account (IAM Identity Center) in `us-east-1`.
- A Qdrant Cloud free cluster in AWS `us-east-1`, with its URL and API key.
- Terraform ≥ 1.10 (`brew install hashicorp/tap/terraform`).
