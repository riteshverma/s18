# Local Qdrant Reindex and Verification

This rollout keeps FAISS artifacts intact and mirrors reindex writes into Qdrant when `ingest.vector_store.provider` is set to `qdrant`.

## Reindex flow

1. Set provider to Qdrant in config (`ingest.vector_store.provider=qdrant`).
2. Trigger reindex through API:
  - `POST /rag/reindex?force=true` for full rebuild
  - `POST /rag/reindex?path=docs/some-file.md` for scoped rebuild
3. During `process_documents`, each successfully indexed file is:
  - written to FAISS metadata/index (legacy kept intact)
  - mirrored into Qdrant via `delete_by_doc(rel_path)` + `upsert(...)`

## Smoke verification

Use these commands from the repo root:

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:6333/' -UseBasicParsing
```

```powershell
uv run python -c "from integrations.vectors import Chunk, get_vector_store; tenant={'tenant_id':'default','tenant_tier':'starter','data_region':'in'}; s=get_vector_store(tenant); s.ensure_index(dimension=3); s.delete_by_doc('qdrant-smoke-doc',tenant_id='default'); s.upsert([Chunk(chunk_id='qdrant-smoke-1',doc_id='qdrant-smoke-doc',text='qdrant smoke test chunk',embedding=[0.1,0.2,0.3],tenant_id='default',integration_id='rag_local',source_uri='docs/smoke.md',metadata={'doc':'docs/smoke.md','page':1})]); h=s.query(embedding=[0.1,0.2,0.3],text='smoke',k=1,filters={'tenant_id':'default'}); print({'provider':s.provider,'hits':[(x.chunk_id,x.doc_id,x.source_uri) for x in h],'count':s.stats().get('document_count')})"
```

Expected: provider `qdrant`, at least one hit with `qdrant-smoke-1`, and non-null collection count.