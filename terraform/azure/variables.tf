variable "resource_group_name" {
  type        = string
  description = "Resource group for S18 cloud-ingest resources."
}

variable "location" {
  type        = string
  description = "Azure region for storage/search resources."
  default     = "eastus"
}

variable "storage_account_name" {
  type        = string
  description = "Globally unique storage account name."
}

variable "blob_container_name" {
  type        = string
  description = "Blob container used by ingest object-store."
  default     = "s18-ingest"
}

variable "search_service_name" {
  type        = string
  description = "Azure AI Search service name."
}

variable "search_sku" {
  type        = string
  description = "Azure AI Search sku."
  default     = "basic"
}
