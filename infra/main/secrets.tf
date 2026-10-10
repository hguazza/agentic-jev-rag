# Terraform creates the secrets, never their values: infra/scripts/put-secrets.sh sets them from .env.aws.
locals {
  secret_names = [
    "OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY", "RECRUITER_API_KEY",
    "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET",
  ]
}

resource "aws_secretsmanager_secret" "app" {
  for_each                = toset(local.secret_names)
  name                    = "${var.project}/${each.key}"
  recovery_window_in_days = 0
}
