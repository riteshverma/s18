variable "region" {
  type        = string
  description = "AWS region."
  default     = "us-east-1"
}

variable "s3_bucket_name" {
  type        = string
  description = "S3 bucket for ingest objects."
}

variable "kms_key_arn" {
  type        = string
  description = "KMS key ARN used for S3 SSE-KMS."
}

variable "aoss_collection_name" {
  type        = string
  description = "OpenSearch Serverless vector collection name."
  default     = "s18-rag-vector"
}

variable "bedrock_kb_id" {
  type        = string
  description = "Optional Bedrock Knowledge Base ID if using managed retrieval."
  default     = ""
}
