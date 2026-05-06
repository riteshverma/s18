from integrations.vectors import factory as vector_factory


def test_vector_factory_routes_to_qdrant_shared(monkeypatch):
    fake_settings = {
        "tenancy": {"growth_routing_enabled": False},
        "ingest": {
            "vector_store": {
                "provider": "qdrant",
                "qdrant": {
                    "url": "http://127.0.0.1:6333",
                    "collection": "s18-rag-{namespace}",
                    "distance": "Cosine",
                },
                "tenant_overrides": {},
            }
        },
    }
    monkeypatch.setattr(vector_factory, "load_settings", lambda: fake_settings)
    store = vector_factory.get_vector_store({"tenant_id": "acme", "tenant_tier": "starter", "data_region": "in"})
    assert store.provider == "qdrant"
    assert store.collection == "s18-rag-shared"


def test_vector_factory_routes_to_qdrant_growth_namespace(monkeypatch):
    fake_settings = {
        "tenancy": {"growth_routing_enabled": True},
        "ingest": {
            "vector_store": {
                "provider": "qdrant",
                "qdrant": {
                    "url": "http://127.0.0.1:6333",
                    "collection": "s18-rag-{namespace}",
                },
                "tenant_overrides": {},
            }
        },
    }
    monkeypatch.setattr(vector_factory, "load_settings", lambda: fake_settings)
    store = vector_factory.get_vector_store({"tenant_id": "acme", "tenant_tier": "growth", "data_region": "us"})
    assert store.provider == "qdrant"
    assert store.collection == "s18-rag-acme__us"
