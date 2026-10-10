# CloudFront terminates HTTPS on its default domain and reaches the internal ALB through a VPC origin,
# so nothing in the VPC has a public address. WAF rate-limits each client IP.

resource "aws_security_group" "alb" {
  name   = "${var.project}-alb"
  vpc_id = aws_vpc.main.id
}

# CloudFront's VPC origin connects from interfaces inside the VPC.
resource "aws_vpc_security_group_ingress_rule" "alb_from_vpc" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = aws_vpc.main.cidr_block
  ip_protocol       = "tcp"
  from_port         = 80
  to_port           = 80
}

resource "aws_vpc_security_group_egress_rule" "alb_to_recruiter" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = aws_security_group.task["recruiter-api"].id
  ip_protocol                  = "tcp"
  from_port                    = 8001
  to_port                      = 8001
}

resource "aws_lb" "recruiter" {
  name                       = "${var.project}-recruiter"
  internal                   = true
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.private[*].id
  idle_timeout               = 120
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "recruiter" {
  name                 = "${var.project}-recruiter"
  port                 = 8001
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.recruiter.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.recruiter.arn
  }
}

resource "aws_cloudfront_vpc_origin" "recruiter" {
  vpc_origin_endpoint_config {
    name                   = "${var.project}-recruiter"
    arn                    = aws_lb.recruiter.arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "http-only"
    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }
}

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}

resource "aws_wafv2_web_acl" "api" {
  name  = "${var.project}-api"
  scope = "CLOUDFRONT"
  default_action {
    allow {}
  }
  rule {
    name     = "rate-limit-per-ip"
    priority = 0
    action {
      block {}
    }
    statement {
      rate_based_statement {
        limit              = 300 # requests per 5 minutes per IP: ample for an interview, fatal for a spend attack
        aggregate_key_type = "IP"
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "rate-limit-per-ip"
      sampled_requests_enabled   = true
    }
  }
  rule {
    name     = "aws-common"
    priority = 1
    override_action {
      none {}
    }
    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
        # Interview answers are free text, and prompt-injection attempts (which the Turn check redirects) look
        # like XSS or path traversal; long answers pass 8 KB. Count these body rules instead of blocking them.
        dynamic "rule_action_override" {
          for_each = ["SizeRestrictions_BODY", "CrossSiteScripting_BODY", "GenericLFI_BODY", "GenericRFI_BODY"]
          content {
            name = rule_action_override.value
            action_to_use {
              count {}
            }
          }
        }
      }
    }
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "aws-common"
      sampled_requests_enabled   = true
    }
  }
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project}-api"
    sampled_requests_enabled   = true
  }
}

resource "aws_cloudfront_distribution" "api" {
  enabled     = true
  comment     = "SwiftAI Recruiter API"
  price_class = "PriceClass_100"
  web_acl_id  = aws_wafv2_web_acl.api.arn

  origin {
    origin_id   = "recruiter-alb"
    domain_name = aws_lb.recruiter.dns_name
    vpc_origin_config {
      vpc_origin_id       = aws_cloudfront_vpc_origin.recruiter.id
      origin_read_timeout = 60
    }
  }

  default_cache_behavior {
    target_origin_id         = "recruiter-alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
