#!/usr/bin/env bash
# Copy the secret values the services need from .env.aws (git-ignored) into Secrets Manager.
# Terraform creates the secrets; their values never pass through Terraform state.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f .env.aws ] || { echo "Create .env.aws from .env.aws.example first." >&2; exit 1; }
set -a; source .env.aws; set +a

for name in OPEN_ROUTER_API_TOKEN QDRANT_API_KEY RECRUITER_API_KEY LIVEKIT_URL LIVEKIT_API_KEY LIVEKIT_API_SECRET; do
  value="${!name:?$name is not set in .env.aws}"
  aws secretsmanager put-secret-value --secret-id "swiftai/$name" --secret-string "$value" >/dev/null
  echo "set swiftai/$name"
done
