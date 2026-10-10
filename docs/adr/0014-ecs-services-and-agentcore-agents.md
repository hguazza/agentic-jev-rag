# Services and interviews on ECS Fargate, the recruitment run on AgentCore Runtime

On AWS, each part of the system runs on the runtime that fits its shape. The RAG API is a stateless FastAPI microservice and the Voice worker holds a long-lived outbound connection to LiveKit, so both run on ECS Fargate. The Recruiter API runs on ECS as the public front door, keeping its typed REST contract and auth, and executes the interview graph in-process: every spoken turn waits on that graph, and moving it to AgentCore would add a hop per turn and a microVM cold start on the first turn and after 15 idle minutes, the latency Jev on the hot path exists to remove.

The recruitment run is the other shape: about three minutes, five agents, mostly waiting on models, started and then polled. That is what Bedrock AgentCore Runtime is for: asynchronous jobs of up to 8 hours in isolated microVMs, independent of API deploys, billed for active CPU rather than wall-clock time. In an optional second phase, the run moves there behind a `RunExecutor` seam in the Recruiter API, with a local and an AgentCore adapter; `/runs` does not change.

Checkpoints move from memory to DynamoDB for both graphs, so interviews survive deploys and task loss and the Recruiter API scales past one task, and the API reads run status from the checkpoint without calling the agent. Compose remains the local runtime (ADR 0008).

## Considered Options

- **Everything on ECS**: phase 1, and complete on its own. Its one weak spot is a run executing on a background thread of the API task that accepted it, which a deploy can interrupt; phase 2 removes that.
- **All Recruiter graphs on AgentCore, interviews included**: rejected. It costs latency on every voice turn and adds nothing: with DynamoDB checkpoints the in-process interview graph is already durable and scalable.
- **Everything on AgentCore**: rejected. The RAG API is not an agent, AgentCore's single `/invocations` endpoint would replace the Recruiter API's typed contract and its tests, and the Voice worker's dispatch model does not fit request-driven invocation.
- **AgentCore Memory as the checkpointer**: deferred. DynamoDB serves both runtimes and stays inspectable; AgentCore Memory is the place for long-term memory across interviews if that is added.
- **Self-hosted Qdrant on ECS with EFS**: rejected for Qdrant Cloud; running a stateful database is not what this project shows.
