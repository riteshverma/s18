"""Mock-only tests for the GCS ObjectStore backend."""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_fake_gcs(monkeypatch):
    class FakeBlob:
        def __init__(self, name, bucket_store):
            self.name = name
            self._store = bucket_store
            self.metadata = {}
            self.content_type = None
            self.size = 0
            self.updated = None
            self.etag = "etag"

        def upload_from_string(self, data, content_type=None):
            self._store[self.name] = {
                "data": data,
                "metadata": dict(self.metadata or {}),
                "content_type": content_type,
            }
            self.size = len(data)
            self.content_type = content_type

        def download_as_bytes(self):
            return self._store[self.name]["data"]

        def generate_signed_url(self, **_kwargs):
            return f"https://signed.local/{self.name}"

        def delete(self):
            self._store.pop(self.name, None)

    class FakeBucket:
        def __init__(self):
            self.store = {}

        def blob(self, name):
            b = FakeBlob(name, self.store)
            if name in self.store:
                b.metadata = dict(self.store[name].get("metadata") or {})
                b.content_type = self.store[name].get("content_type")
                b.size = len(self.store[name]["data"])
            return b

        def get_blob(self, name):
            if name not in self.store:
                return None
            return self.blob(name)

    class FakeClient:
        _buckets = {}

        def __init__(self, project=None):
            self.project = project

        def bucket(self, name):
            if name not in self._buckets:
                self._buckets[name] = FakeBucket()
            return self._buckets[name]

        def list_blobs(self, bucket, prefix=None):
            b = self.bucket(bucket)
            for name, val in b.store.items():
                if prefix and not name.startswith(prefix):
                    continue
                blob = b.blob(name)
                blob.metadata = val.get("metadata") or {}
                blob.size = len(val["data"])
                blob.content_type = val.get("content_type")
                yield blob

    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    storage_module = types.ModuleType("google.cloud.storage")
    storage_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)


def test_gcs_put_get_list_stat_and_delete(monkeypatch):
    _install_fake_gcs(monkeypatch)
    from integrations.storage.gcs import GcsObjectStore

    store = GcsObjectStore(bucket="acme-ingest", namespace="tenant-a", project="acme-proj")
    ref = store.put("docs/a.txt", b"hello", content_type="text/plain", metadata={"x": "1"})
    assert ref.provider == "gcs"
    assert ref.uri.startswith("gs://acme-ingest/tenant-a/docs/a.txt")
    assert store.get(ref.uri) == b"hello"

    listed = list(store.list("docs"))
    assert len(listed) == 1
    assert listed[0].sha256

    info = store.stat(ref.uri)
    assert info["size"] == 5
    assert info["metadata"]["x"] == "1"
    assert store.presign(ref.uri).startswith("https://signed.local/")

    store.delete(ref.uri)
    assert list(store.list("docs")) == []


def test_storage_factory_routes_to_gcs(monkeypatch):
    _install_fake_gcs(monkeypatch)
    from integrations.storage import factory as storage_factory

    fake_settings = {
        "tenancy": {"growth_routing_enabled": False},
        "ingest": {
            "object_store": {
                "provider": "gcs",
                "gcs": {"bucket": "acme-ingest", "project": "acme-proj"},
                "tenant_overrides": {},
            }
        },
    }
    monkeypatch.setattr(storage_factory, "load_settings", lambda: fake_settings)
    store = storage_factory.get_object_store({"tenant_id": "acme", "tenant_tier": "starter"})
    assert store.provider == "gcs"
