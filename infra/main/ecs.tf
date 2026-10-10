resource "aws_ecs_cluster" "main" {
  name = var.project
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Names any client in the VPC resolves (Cloud Map DNS, not Service Connect), so AgentCore can use them in phase 2.
resource "aws_service_discovery_private_dns_namespace" "internal" {
  name = "${var.project}.internal"
  vpc  = aws_vpc.main.id
}

locals {
  api_image   = "${aws_ecr_repository.image["api"].repository_url}:${var.image_tag}"
  voice_image = "${aws_ecr_repository.image["voice"].repository_url}:${var.image_tag}"
  internal    = "${var.project}.internal"

  # Every task the stack runs; all but ingest are long-running services.
  tasks = {
    "rag-api" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = null
      command = ["uvicorn", "agentic_jev_rag.rag.api:create_default_rag_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
      env     = { QDRANT_URL = var.qdrant_url, SYSTEM_1 = var.system_1 }
      secrets = ["OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY"]
    }
    # Keep-alive above the ALB's 120 s idle timeout, so the ALB never reuses a connection uvicorn closed (502s).
    "recruiter-api" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = aws_iam_role.recruiter.arn
      command = ["uvicorn", "agentic_jev_rag.recruiter.api:create_default_recruiter_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--timeout-keep-alive", "125"]
      env = {
        AWS_REGION       = var.region
        RAG_URL          = "http://rag-api.${local.internal}:8000"
        SYSTEM_1         = var.system_1
        PERSISTENCE      = "dynamodb"
        CHECKPOINT_TABLE = aws_dynamodb_table.checkpoints.name
        RUNS_TABLE       = aws_dynamodb_table.runs.name
      }
      secrets = ["OPEN_ROUTER_API_TOKEN", "RECRUITER_API_KEY"]
    }
    "voice-worker" = {
      image   = local.voice_image
      cpu     = 1024
      memory  = 2048
      role    = null
      command = ["python", "-m", "agentic_jev_rag.voice", "start"]
      env     = { RECRUITER_URL = "http://recruiter-api.${local.internal}:8001" }
      secrets = ["OPEN_ROUTER_API_TOKEN", "RECRUITER_API_KEY", "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
    }
    "ingest" = {
      image   = local.api_image
      cpu     = 512
      memory  = 1024
      role    = null
      command = ["python", "-m", "agentic_jev_rag.rag.ingest"]
      env     = { QDRANT_URL = var.qdrant_url }
      secrets = ["OPEN_ROUTER_API_TOKEN", "QDRANT_API_KEY"]
    }
  }

  # Services with an HTTP port get a health check and a Cloud Map name.
  ports = { "rag-api" = 8000, "recruiter-api" = 8001 }

  scaling = {
    "rag-api"       = { min = 2, max = 4 }
    "recruiter-api" = { min = 2, max = 4 }
    "voice-worker"  = { min = 1, max = 1 }
  }
}

resource "aws_cloudwatch_log_group" "task" {
  for_each          = local.tasks
  name              = "/ecs/${var.project}/${each.key}"
  retention_in_days = 14
}

resource "aws_security_group" "task" {
  for_each = local.tasks
  name     = "${var.project}-${each.key}"
  vpc_id   = aws_vpc.main.id
}

# Every task calls the internet: OpenRouter, Qdrant Cloud, LiveKit (through the NAT gateway).
resource "aws_vpc_security_group_egress_rule" "task_internet" {
  for_each          = aws_security_group.task
  security_group_id = each.value.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}

resource "aws_vpc_security_group_ingress_rule" "recruiter_from_alb" {
  security_group_id            = aws_security_group.task["recruiter-api"].id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_vpc_security_group_ingress_rule" "recruiter_from_voice" {
  security_group_id            = aws_security_group.task["recruiter-api"].id
  referenced_security_group_id = aws_security_group.task["voice-worker"].id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_vpc_security_group_ingress_rule" "rag_from_recruiter" {
  security_group_id            = aws_security_group.task["rag-api"].id
  referenced_security_group_id = aws_security_group.task["recruiter-api"].id
  ip_protocol                  = "tcp"
  from_port                    = 8000
  to_port                      = 8000
}

resource "aws_ecs_task_definition" "task" {
  for_each                 = local.tasks
  family                   = "${var.project}-${each.key}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = each.value.cpu
  memory                   = each.value.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = each.value.role
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }
  container_definitions = jsonencode([{
    name         = each.key
    image        = each.value.image
    essential    = true
    command      = each.value.command
    stopTimeout  = 120
    environment  = [for name, value in each.value.env : { name = name, value = value }]
    secrets      = [for name in each.value.secrets : { name = name, valueFrom = aws_secretsmanager_secret.app[name].arn }]
    portMappings = [for port in try([local.ports[each.key]], []) : { containerPort = port, protocol = "tcp" }]
    healthCheck = contains(keys(local.ports), each.key) ? {
      command     = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:${local.ports[each.key]}/health')\""]
      interval    = 15
      timeout     = 5
      retries     = 3
      startPeriod = 30
    } : null
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.task[each.key].name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = each.key
      }
    }
  }])
}

resource "aws_service_discovery_service" "api" {
  for_each = local.ports
  name     = each.key
  dns_config {
    namespace_id   = aws_service_discovery_private_dns_namespace.internal.id
    routing_policy = "MULTIVALUE"
    dns_records {
      type = "A"
      ttl  = 10
    }
  }
}

resource "aws_ecs_service" "service" {
  for_each        = local.scaling
  name            = each.key
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.task[each.key].arn
  desired_count   = var.running ? each.value.min : 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.task[each.key].id]
    assign_public_ip = false
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  dynamic "service_registries" {
    for_each = contains(keys(local.ports), each.key) ? [each.key] : []
    content {
      registry_arn = aws_service_discovery_service.api[service_registries.value].arn
    }
  }

  dynamic "load_balancer" {
    for_each = each.key == "recruiter-api" ? [8001] : []
    content {
      target_group_arn = aws_lb_target_group.recruiter.arn
      container_name   = each.key
      container_port   = load_balancer.value
    }
  }

  health_check_grace_period_seconds = each.key == "recruiter-api" ? 60 : null

  # Not ignored on purpose: an apply resets the count to the minimum (or to 0 with running = false), because
  # Application Auto Scaling only applies new bounds when a scaling activity fires. Autoscaling adjusts from there.

  depends_on = [aws_lb_listener.http]
}

resource "aws_appautoscaling_target" "service" {
  for_each           = local.scaling
  service_namespace  = "ecs"
  scalable_dimension = "ecs:service:DesiredCount"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.service[each.key].name}"
  min_capacity       = var.running ? each.value.min : 0
  max_capacity       = var.running ? each.value.max : 0
}

resource "aws_appautoscaling_policy" "rag_cpu" {
  name               = "${var.project}-rag-api-cpu"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.service["rag-api"].service_namespace
  scalable_dimension = aws_appautoscaling_target.service["rag-api"].scalable_dimension
  resource_id        = aws_appautoscaling_target.service["rag-api"].resource_id
  target_tracking_scaling_policy_configuration {
    target_value = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}

resource "aws_appautoscaling_policy" "recruiter_requests" {
  name               = "${var.project}-recruiter-api-requests"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.service["recruiter-api"].service_namespace
  scalable_dimension = aws_appautoscaling_target.service["recruiter-api"].scalable_dimension
  resource_id        = aws_appautoscaling_target.service["recruiter-api"].resource_id
  target_tracking_scaling_policy_configuration {
    target_value = 50
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.recruiter.arn_suffix}/${aws_lb_target_group.recruiter.arn_suffix}"
    }
  }
}
