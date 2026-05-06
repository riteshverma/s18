import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from integrations.vectors.base import SearchHit
from routers import rag as rag_router


class RagSearchRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_qdrant_provider_uses_local_hybrid_mcp_path(self):
        fake_settings = {
            "ingest": {"vector_store": {"provider": "qdrant", "tenant_overrides": {}}},
            "tenancy": {},
            "rag": {"top_k": 3},
        }
        fake_user = {"sub": "user-1"}
        fake_tenant = {"tenant_id": "default", "tenant_tier": "starter", "data_region": "in"}

        class _FakeMcp:
            async def call_tool(self, server, tool, args):
                self.called = (server, tool, args)
                return SimpleNamespace(content=[{"text": "['hello\\n[Source: docs/a.md p2]']"}])

        fake_mcp = _FakeMcp()
        with patch.object(rag_router, "settings", fake_settings), patch.object(
            rag_router, "resolve_tenant_context", return_value=fake_tenant
        ), patch.object(rag_router, "multi_mcp", fake_mcp):
            out = await rag_router.rag_search("hello", user=fake_user)

        self.assertEqual(out["status"], "success")
        self.assertEqual(out["vector_provider"], "qdrant")
        self.assertEqual(fake_mcp.called[1], "search_stored_documents_rag")
        self.assertEqual(out["results"][0]["source"], "docs/a.md")
        self.assertEqual(out["results"][0]["page"], 2)

    async def test_cloud_provider_keeps_vector_store_path(self):
        fake_settings = {
            "ingest": {"vector_store": {"provider": "azure_ai_search", "tenant_overrides": {}}},
            "tenancy": {},
            "rag": {"top_k": 3},
        }
        fake_user = {"sub": "user-1"}
        fake_tenant = {"tenant_id": "default", "tenant_tier": "starter", "data_region": "in"}
        fake_rerank_cfg = SimpleNamespace(enabled=False, top_k=None, candidate_k=10)

        class _FakeStore:
            provider = "azure_ai_search"

            def query(self, **kwargs):
                self.kwargs = kwargs
                return [
                    SearchHit(
                        chunk_id="c1",
                        doc_id="d1",
                        text="cloud hit",
                        score=0.8,
                        source_uri="docs/cloud.md",
                        metadata={"page": 4},
                    )
                ]

        fake_store = _FakeStore()
        with patch.object(rag_router, "settings", fake_settings), patch.object(
            rag_router, "resolve_tenant_context", return_value=fake_tenant
        ), patch("core.embedding.get_normalized_embedding", return_value=np.array([0.1, 0.2, 0.3])), patch.object(
            rag_router, "get_vector_store", return_value=fake_store
        ), patch.object(
            rag_router, "load_rerank_config", return_value=fake_rerank_cfg
        ):
            out = await rag_router.rag_search("cloud", user=fake_user)

        self.assertEqual(out["status"], "success")
        self.assertEqual(out["vector_provider"], "azure_ai_search")
        self.assertEqual(out["results"][0]["source"], "docs/cloud.md")
        self.assertEqual(fake_store.kwargs["filters"]["tenant_id"], "default")


if __name__ == "__main__":
    unittest.main()
