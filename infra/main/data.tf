# Interview checkpoints, with the key schema langgraph-checkpoint-aws's DynamoDBSaver expects.
resource "aws_dynamodb_table" "checkpoints" {
  name         = "${var.project}-checkpoints"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"
  attribute {
    name = "PK"
    type = "S"
  }
  attribute {
    name = "SK"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled = true
  }
}

# Recruitment run records: status, report and error per run_id.
resource "aws_dynamodb_table" "runs" {
  name         = "${var.project}-runs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  attribute {
    name = "run_id"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
  point_in_time_recovery {
    enabled = true
  }
  server_side_encryption {
    enabled = true
  }
}
