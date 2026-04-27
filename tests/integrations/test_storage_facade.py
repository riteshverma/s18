"""Smoke tests for the ObjectStore facade.

Local FS path is exercised end-to-end; the Azure / AWS paths are constructed
without invoking real cloud SDKs (lazy imports keep the test deps minimal).
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.storage.local_fs import LocalFsObjectStore


def test_local_fs_put_get_roundtrip(tmp_path: Path):
    store = LocalFsObjectStore(root=tmp_path, namespace="acme")
    ref = store.put("hello/world.txt", b"hi", content_type="text/plain", metadata={"k": "v"})

    assert ref.provider == "local_fs"
    assert ref.size == 2
    assert ref.sha256
    assert ref.metadata["k"] == "v"
    assert ref.metadata["content_type"] == "text/plain"

    assert store.get(ref.uri) == b"hi"
    info = store.stat(ref.uri)
    assert info["size"] == 2
    assert info["sha256"] == ref.sha256


def test_local_fs_list_skips_meta_files(tmp_path: Path):
    store = LocalFsObjectStore(root=tmp_path, namespace="acme")
    store.put("a/1.txt", b"one")
    store.put("a/2.txt", b"two")

    refs = list(store.list("a"))
    keys = sorted(r.key for r in refs)
    assert keys == ["a/1.txt", "a/2.txt"]


def test_local_fs_blocks_path_escape(tmp_path: Path):
    store = LocalFsObjectStore(root=tmp_path, namespace="acme")
    # `..` segments are scrubbed before resolution; result must land inside
    # the namespace root.
    ref = store.put("../escape.txt", b"x")
    namespace_root = (tmp_path / "acme").resolve()
    resolved = Path(store._uri_to_path(ref.uri))
    assert namespace_root in resolved.parents or resolved.parent == namespace_root


def test_factory_picks_tenant_override(monkeypatch, tmp_path: Path):
    from integrations.storage import factory as storage_factory

    fake_settings = {
        "tenancy": {"growth_routing_enabled": False},
        "ingest": {
            "object_store": {
                "provider": "local_fs",
                "local_fs": {"root": str(tmp_path)},
                "tenant_overrides": {"acme": "local_fs"},
            }
        },
    }
    monkeypatch.setattr(storage_factory, "load_settings", lambda: fake_settings)

    store = storage_factory.get_object_store({"tenant_id": "acme", "tenant_tier": "starter"})
    assert store.provider == "local_fs"
    ref = store.put("k.txt", b"data")
    assert store.get(ref.uri) == b"data"


def test_azure_blob_backend_constructs_without_sdk():
    """The Azure facade must instantiate without azure-storage-blob installed."""
    from integrations.storage.azure_blob import AzureBlobObjectStore

    store = AzureBlobObjectStore(
        account_url="https://acme.blob.core.windows.net",
        container="ingest",
        namespace="acme",
    )
    assert store.provider == "azure_blob"
    assert store.container == "ingest"
    assert store.namespace == "acme"


def test_aws_s3_backend_constructs_without_sdk():
    """The AWS facade must instantiate without boto3 installed."""
    from integrations.storage.aws_s3 import AwsS3ObjectStore

    store = AwsS3ObjectStore(bucket="acme-bucket", region="us-east-1", namespace="acme")
    assert store.provider == "aws_s3"
    assert store.bucket == "acme-bucket"
    assert store.namespace == "acme"
