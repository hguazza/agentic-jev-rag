variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "agentic-recruiter"
}

variable "image_tag" {
  description = "Git SHA the api and voice images are tagged with in ECR."
  type        = string
}

variable "running" {
  description = "false scales every service to zero and removes the NAT gateway, for between demos."
  type        = bool
  default     = true
}

variable "alert_email" {
  description = "Receives alarm and budget emails (confirm the SNS subscription email once)."
  type        = string
}

variable "qdrant_url" {
  description = "The Qdrant Cloud cluster URL (its API key is in Secrets Manager)."
  type        = string
}

variable "system_1" {
  description = "Default System 1 decider: llm or jev."
  type        = string
  default     = "llm"
}

variable "monthly_budget_usd" {
  type    = number
  default = 60
}
