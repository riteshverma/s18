output "aws_region" {
  value       = var.region
  description = "Set AWS_REGION in configmap."
}

output "s3_bucket" {
  value       = aws_s3_bucket.ingest.bucket
  description = "Set S3_BUCKET in configmap."
}

output "s3_kms_key" {
  value       = var.kms_key_arn
  description = "Set S3_KMS_KEY_ID in configmap."
}

output "opensearch_endpoint_hint" {
  value       = "Populate OPENSEARCH_ENDPOINT from the OpenSearch domain/collection endpoint after provisioning policies."
  description = "OpenSearch endpoint wiring hint."
}

output "bedrock_kb_id" {
  value       = var.bedrock_kb_id
  description = "Set BEDROCK_KB_ID if Bedrock Knowledge Base is used."
}
