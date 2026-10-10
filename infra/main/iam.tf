data "aws_iam_policy_document" "ecs_tasks_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Pulls images, writes logs and injects secrets for every task.
resource "aws_iam_role" "execution" {
  name               = "${var.project}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [for secret in aws_secretsmanager_secret.app : secret.arn]
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# Only the Recruiter API calls AWS at runtime: its checkpoints and run records.
resource "aws_iam_role" "recruiter" {
  name               = "${var.project}-recruiter-api"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_trust.json
}

data "aws_iam_policy_document" "recruiter" {
  statement {
    actions = [
      "dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:Query",
      "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem",
    ]
    resources = [aws_dynamodb_table.checkpoints.arn, aws_dynamodb_table.runs.arn]
  }
}

resource "aws_iam_role_policy" "recruiter" {
  role   = aws_iam_role.recruiter.id
  policy = data.aws_iam_policy_document.recruiter.json
}
