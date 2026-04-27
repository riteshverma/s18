"""Mock-only tests for Vertex AI Vector Search backend."""

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from integrations.vectors import Chunk


def _install_fake_vertex(monkeypatch):
    class Restriction:
        def __init__(self, namespace=None, allow_list=None):
            self.namespace = namespace
            self.allow_list = allow_list or []

    class CrowdingTag:
        def __init__(self, crowding_attribute=None):
            self.crowding_attribute = crowding_attribute

    class IndexDatapoint:
        def __init__(
            self,
            datapoint_id=None,
            feature_vector=None,
            restricts=None,
            crowding_tag=None,
        ):
            self.datapoint_id = datapoint_id
            self.feature_vector = feature_vector or []
            self.restricts = restricts or []
            self.crowding_tag = crowding_tag

    IndexDatapoint.Restriction = Restriction
    IndexDatapoint.CrowdingTag = CrowdingTag

    class Query:
        def __init__(self, datapoint=None, neighbor_count=5):
            self.datapoint = datapoint
            self.neighbor_count = neighbor_count

    class FindNeighborsRequest:
        def __init__(self, index_endpoint=None, deployed_index_id=None, queries=None, return_full_datapoint=False):
            self.index_endpoint = index_endpoint
            self.deployed_index_id = deployed_index_id
            self.queries = queries or []
            self.return_full_datapoint = return_full_datapoint

    FindNeighborsRequest.Query = Query

    class FakeNeighbor:
        def __init__(self, idx, dist):
            self.datapoint = types.SimpleNamespace(datapoint_id=idx)
            self.distance = dist

    class FakeNearestNeighbors:
        def __init__(self):
            self.neighbors = [FakeNeighbor("c1", 0.1), FakeNeighbor("c2", 0.3)]

    class FakeMatchServiceClient:
        def __init__(self, **_kwargs):
            self.calls = []

        def find_neighbors(self, request):
            self.calls.append(request)
            return types.SimpleNamespace(nearest_neighbors=[FakeNearestNeighbors()])

    class FakeIndexServiceClient:
        def __init__(self, **_kwargs):
            self.upserted = []

        def get_index_endpoint(self, name):
            return types.SimpleNamespace(name=name, deployed_indexes=[1])

        def get_index(self, name):
            return types.SimpleNamespace(name=name)

        def upsert_datapoints(self, index, datapoints):
            self.upserted.append((index, datapoints))

    aiplatform_v1 = types.ModuleType("google.cloud.aiplatform_v1")
    aiplatform_v1.IndexServiceClient = FakeIndexServiceClient
    aiplatform_v1.MatchServiceClient = FakeMatchServiceClient
    aiplatform_v1.IndexDatapoint = IndexDatapoint
    aiplatform_v1.FindNeighborsRequest = FindNeighborsRequest

    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform_v1", aiplatform_v1)


def test_vertex_store_upsert_and_query(monkeypatch):
    _install_fake_vertex(monkeypatch)
    from integrations.vectors.vertex_ai_vector_search import VertexAiVectorSearchStore

    store = VertexAiVectorSearchStore(
        project="acme-proj",
        location="us-central1",
        index_endpoint_id="123",
        deployed_index_id="dep-1",
        index_id="idx-1",
    )
    store.ensure_index(dimension=4)
    inserted = store.upsert(
        [
            Chunk(
                chunk_id="c1",
                doc_id="d1",
                text="hello",
                embedding=[0.1, 0.2, 0.3, 0.4],
                tenant_id="acme",
                integration_id="powerapps",
            )
        ]
    )
    assert inserted == 1
    hits = store.query(embedding=[0.1, 0.2, 0.3, 0.4], k=2, filters={"tenant_id": "acme"})
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert store.stats()["provider"] == "vertex_ai_vector_search"


def test_vector_factory_routes_to_vertex(monkeypatch):
    _install_fake_vertex(monkeypatch)
    from integrations.vectors import factory as vector_factory

    fake_settings = {
        "tenancy": {"growth_routing_enabled": False},
        "ingest": {
            "vector_store": {
                "provider": "vertex_ai_vector_search",
                "vertex_ai_vector_search": {
                    "project": "acme-proj",
                    "location": "us-central1",
                    "index_endpoint_id": "123",
                    "deployed_index_id": "dep-1",
                    "index_id": "idx-1",
                },
                "tenant_overrides": {},
            }
        },
    }
    monkeypatch.setattr(vector_factory, "load_settings", lambda: fake_settings)
    store = vector_factory.get_vector_store({"tenant_id": "acme", "tenant_tier": "starter"})
    assert store.provider == "vertex_ai_vector_search"
