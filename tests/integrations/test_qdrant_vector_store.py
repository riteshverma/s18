from types import SimpleNamespace

from integrations.vectors.base import Chunk
from integrations.vectors.qdrant_store import QdrantVectorStore


class _FakeQdrantClient:
    def __init__(self):
        self.created_collection = None
        self.upsert_calls = []
        self.search_calls = []
        self.scroll_calls = []
        self.deleted = []
        self.collection_exists_value = False

    def collection_exists(self, collection_name):
        return self.collection_exists_value

    def get_collection(self, collection_name):
        vectors = SimpleNamespace(size=3)
        params = SimpleNamespace(vectors=vectors)
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def create_collection(self, collection_name, vectors_config):
        self.created_collection = (collection_name, vectors_config)

    def upsert(self, collection_name, points, wait):
        self.upsert_calls.append((collection_name, points, wait))

    def search(self, collection_name, query_vector, query_filter, limit, with_payload):
        self.search_calls.append((collection_name, query_vector, query_filter, limit, with_payload))
        return [
            SimpleNamespace(
                id="chunk-1",
                score=0.91,
                payload={
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "chunk": "hello world",
                    "doc": "docs/a.md",
                    "source_uri": "docs/a.md",
                    "metadata": {"page": 3, "tenant_id": "default"},
                },
            )
        ]

    def scroll(self, collection_name, **kwargs):
        self.scroll_calls.append((collection_name, kwargs))
        filt = kwargs.get("scroll_filter")
        if filt is None:
            return [], None
        return [SimpleNamespace(id="chunk-1", payload={"chunk_id": "chunk-1"})], None

    def delete(self, collection_name, points_selector, wait):
        self.deleted.append((collection_name, points_selector, wait))
        return SimpleNamespace(status="ack")

    def count(self, collection_name, exact):
        return SimpleNamespace(count=7)


def test_qdrant_store_ensure_upsert_query_delete_stats():
    fake = _FakeQdrantClient()
    store = QdrantVectorStore(
        url="http://127.0.0.1:6333",
        collection="s18-rag-shared",
        distance="Cosine",
    )
    store._client = fake

    store.ensure_index(dimension=3)
    assert fake.created_collection is not None
    assert fake.created_collection[0] == "s18-rag-shared"

    inserted = store.upsert(
        [
            Chunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="hello world",
                embedding=[0.1, 0.2, 0.3],
                tenant_id="default",
                integration_id="powerapps",
                source_uri="docs/a.md",
                metadata={"doc": "docs/a.md", "page": 3},
            )
        ]
    )
    assert inserted == 1
    assert len(fake.upsert_calls) == 1

    hits = store.query(embedding=[0.1, 0.2, 0.3], text="hello", k=3, filters={"tenant_id": "default"})
    assert len(hits) == 1
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].metadata.get("page") == 3

    removed = store.delete_by_doc("doc-1", tenant_id="default")
    assert removed == 1
    assert len(fake.deleted) == 1

    stats = store.stats()
    assert stats["provider"] == "qdrant"
    assert stats["document_count"] == 7


def test_qdrant_store_export_metadata():
    fake = _FakeQdrantClient()

    def _scroll(collection_name, **kwargs):
        return [
            SimpleNamespace(
                id="chunk-2",
                payload={
                    "chunk_id": "chunk-2",
                    "doc_id": "doc-2",
                    "doc": "docs/b.md",
                    "chunk": "medical summary",
                    "page": 9,
                    "tenant_id": "default",
                    "metadata": {"category": "clinical"},
                },
            )
        ], None

    fake.scroll = _scroll

    store = QdrantVectorStore(
        url="http://127.0.0.1:6333",
        collection="s18-rag-shared",
    )
    store._client = fake

    metadata = store.export_metadata(tenant_id="default")
    assert len(metadata) == 1
    assert metadata[0]["chunk_id"] == "chunk-2"
    assert metadata[0]["doc"] == "docs/b.md"
    assert metadata[0]["page"] == 9
    assert metadata[0]["category"] == "clinical"
