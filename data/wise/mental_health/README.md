# Mental health reference (RAG)

## DSM-5 PDF

Place the reference PDF here as **`DSM-5_reference.pdf`** (relative to the S18 repo: `data/wise/mental_health/DSM-5_reference.pdf`).

A copy is typically synced from the WISE repo-for-reference location:

`wise-ai/docs/Diagnostic_and_statistical_manual_of_mental_disorders_DSM-5_(_PDFDrive.com_).pdf`

Do not commit copyrighted PDFs to public repos; keep this file **local** or in private storage.

## Index after add or replace

With the API running (`uv run python api.py`), rebuild the FAISS index for this folder:

```http
POST /rag/reindex?path=wise/mental_health&force=false
```

Open `http://localhost:8000/docs` and execute **POST /rag/reindex** with query `path=wise/mental_health` if you prefer Swagger.

For a full rebuild after changing embedding model settings, use `force=true`.

## Retrieval

The MCP tool `search_stored_documents_rag` accepts optional **`doc_path`** to restrict search to this book, e.g. `wise/mental_health/DSM-5_reference.pdf` (see `mcp_servers/README_rag.md`).

Mental health runs that include `[Task: mental_health]` can phrase retriever queries with symptoms plus DSM-5 framing so vector search surfaces the indexed sections.
