# Terraform Scaffolds (Azure, AWS, GCP)

This directory contains operator-facing Terraform scaffolds for provisioning
the cloud data-plane dependencies used by the Power Apps cloud ingest pipeline.

The application runtime itself remains in the existing Kubernetes manifests
under `k8s/`. These Terraform stacks provision cloud services and output values
that map into `k8s/configmap.yaml` and `k8s/secret.yaml`.

## Layout

- `terraform/azure/` - Azure Blob, Azure AI Search, Azure OpenAI wiring.
- `terraform/aws/` - S3, OpenSearch Serverless, Bedrock KB wiring.
- `terraform/gcp/` - GCS, Vertex AI Vector Search, Vertex embeddings wiring.

## Typical workflow

1. Pick one provider module per tenant/workload.
2. Set variables (`terraform.tfvars`) for project/subscription/account names.
3. `terraform init && terraform plan && terraform apply`.
4. Copy stack outputs into:
   - `k8s/configmap.yaml` (`AZURE_*`, `AWS_*`, `VERTEX_*`, etc.)
   - `k8s/secret.yaml` (`AZURE_OPENAI_API_KEY`, cloud service creds if not using workload identity)

## Notes

- These are baseline scaffolds and intentionally conservative.
- Network controls (private endpoints, VPC/PSC, firewall rules) should be
  enabled per your compliance posture.
- Use remote Terraform state and policy checks in production.
