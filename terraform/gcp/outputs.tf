output "google_cloud_project" {
  value       = var.project_id
  description = "Set GOOGLE_CLOUD_PROJECT in configmap."
}

output "gcs_bucket" {
  value       = google_storage_bucket.ingest.name
  description = "Set GCS_BUCKET in configmap."
}

output "vertex_ai_location" {
  value       = var.region
  description = "Set VERTEX_AI_LOCATION in configmap."
}

output "vertex_ai_index_id" {
  value       = var.vertex_index_id
  description = "Set VERTEX_AI_INDEX_ID in configmap."
}

output "vertex_ai_index_endpoint_id" {
  value       = var.vertex_index_endpoint_id
  description = "Set VERTEX_AI_INDEX_ENDPOINT_ID in configmap."
}

output "vertex_ai_deployed_index_id" {
  value       = var.vertex_deployed_index_id
  description = "Set VERTEX_AI_DEPLOYED_INDEX_ID in configmap."
}
