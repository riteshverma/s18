# Power Apps -> S18 RAG ingest

This folder contains the artifacts a Power Platform admin needs to wire up
the S18 RAG ingest pipeline:

| File | What it is |
| ---- | ---------- |
| [`custom_connector_openapi.yaml`](custom_connector_openapi.yaml) | OpenAPI 2.0 (Swagger) spec for the **S18 RAG Ingest** custom connector. Imports directly into Power Apps / Power Automate via *Custom connectors -> New custom connector -> Import an OpenAPI file*. |
| [`power_automate_flow.dataverse.json`](power_automate_flow.dataverse.json) | Reference flow that fires when a Dataverse row is created or updated and posts the row to `/ingest/powerapps`. |
| [`power_automate_flow.sharepoint_files.json`](power_automate_flow.sharepoint_files.json) | Reference flow that fires when a SharePoint document is uploaded and pushes the binary plus list-item metadata to `/ingest/powerapps`. |

## Setup checklist

1. Register an Entra ID app for the connector (`api://s18-rag`) with the
   delegated scope `Ingest.Write`. Grant it to the Power Platform service
   principal that will own the connector.
2. Import `custom_connector_openapi.yaml` and replace `host:` with your S18
   API hostname.
3. In each tenant's `config/integrations/powerapps_<workflow>_v1.json`, pin
   `object_store_provider` and `vector_store_provider` to the cloud they
   want (Azure or AWS).
4. Verify backend selection by calling `GET /ingest/health` from the
   connector's *Test* tab; it returns the resolved providers per tenant.
5. Import the reference flows and edit the parameters (table name, tenant
   id, workflow id) to match the tenant.

## Payload contract

See `IngestEnvelope` in `custom_connector_openapi.yaml` for the canonical
shape; the same structure is consumed by both the JSON and multipart
endpoints.

Files larger than ~8 MB should use the multipart variant
(`POST /ingest/powerapps/files`) since Power Automate base64-inflates inline
attachments.
