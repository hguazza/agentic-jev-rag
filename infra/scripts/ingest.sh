#!/usr/bin/env bash
# Index the committed corpus into Qdrant Cloud with a one-off ECS task and wait for it to finish.
set -euo pipefail
cd "$(dirname "$0")/../main"

cluster=$(terraform output -raw cluster_name)
subnets=$(terraform output -json private_subnet_ids | jq -r 'join(",")')
group=$(terraform output -raw ingest_security_group_id)
task=$(aws ecs run-task --cluster "$cluster" --launch-type FARGATE \
  --task-definition "$(terraform output -raw ingest_task_definition)" \
  --network-configuration "awsvpcConfiguration={subnets=[$subnets],securityGroups=[$group],assignPublicIp=DISABLED}" \
  --query 'tasks[0].taskArn' --output text)
echo "ingest task: $task"

aws ecs wait tasks-stopped --cluster "$cluster" --tasks "$task"
code=$(aws ecs describe-tasks --cluster "$cluster" --tasks "$task" --query 'tasks[0].containers[0].exitCode' --output text)
echo "ingest exit code: $code (logs: /ecs/agentic-recruiter/ingest)"
[ "$code" = "0" ]
