output "api_url" {
  description = "The Recruiter API's public HTTPS URL."
  value       = "https://${aws_cloudfront_distribution.api.domain_name}"
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "ingest_security_group_id" {
  value = aws_security_group.task["ingest"].id
}

output "ingest_task_definition" {
  value = aws_ecs_task_definition.task["ingest"].arn
}
