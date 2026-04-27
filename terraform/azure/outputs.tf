output "azure_storage_account" {
  value       = azurerm_storage_account.this.name
  description = "Set AZURE_STORAGE_ACCOUNT in k8s configmap."
}

output "azure_storage_container" {
  value       = azurerm_storage_container.ingest.name
  description = "Set AZURE_STORAGE_CONTAINER in k8s configmap."
}

output "azure_search_endpoint" {
  value       = "https://${azurerm_search_service.this.name}.search.windows.net"
  description = "Set AZURE_SEARCH_ENDPOINT in k8s configmap."
}

output "azure_openai_hint" {
  value       = "Configure AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_* deployment env vars separately in configmap."
  description = "Operational hint; Azure OpenAI resource itself can be provisioned in a dedicated module."
}
