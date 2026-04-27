"""Cloud-agnostic object storage facade for the Power Apps ingest pipeline.

Tenants choose Azure Blob, AWS S3, or local filesystem at config time; callers
work against the :class:`ObjectStore` Protocol and never import a cloud SDK.

Usage:

    from integrations.storage import get_object_store

    store = get_object_store(tenant_context)
    ref = store.put("invoices/2026-04/inv-1.pdf", pdf_bytes, metadata={...})
    body = store.get(ref.uri)
"""

from integrations.storage.base import ObjectRef, ObjectStore
from integrations.storage.factory import get_object_store

__all__ = ["ObjectRef", "ObjectStore", "get_object_store"]
