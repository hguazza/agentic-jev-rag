# Docker Compose is the runtime, from development onward

All services (Qdrant, and later the FastAPI app) run under Docker Compose, including during development, rather than Qdrant's embedded local mode. This matches the owner's standard practice and the later plan to deploy to a cloud VM, and gives readers re-running the benchmark one setup command.
