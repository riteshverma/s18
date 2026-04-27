variable "project_id" {
  type        = string
  description = "GCP project ID."
}

variable "region" {
  type        = string
  description = "GCP region for GCS and Vertex AI."
  default     = "us-central1"
}

variable "gcs_bucket_name" {
  type        = string
  description = "GCS bucket name used by ingest object-store."
}

variable "vertex_index_id" {
  type        = string
  description = "Vertex AI Vector Search Index ID."
}

variable "vertex_index_endpoint_id" {
  type        = string
  description = "Vertex AI Index Endpoint ID."
}

variable "vertex_deployed_index_id" {
  type        = string
  description = "Vertex AI deployed index ID used by query traffic."
}
