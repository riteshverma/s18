# RAG Router - Handles document management, indexing, and search
import json
import re
from pathlib import Path
from typing import Any, Dict
import ast
import logging
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, StreamingResponse
import hashlib
from PIL import Image
import os
import requests
import io

from shared.state import get_multi_mcp, PROJECT_ROOT
from config.settings_loader import settings
from core.supabase_auth import require_supabase_user
from core.prometheus_metrics import (
    RAG_EMPTY_RESULT_TOTAL,
    RAG_REQUESTS_TOTAL,
    RAG_RESULTS_COUNT,
    RAG_SEARCH_LATENCY_MS,
    elapsed_ms,
    now_ms,
)
from integrations.tenancy import resolve_tenant_context
from integrations.vectors import get_vector_store

router = APIRouter(prefix="/rag", tags=["RAG"])
logger = logging.getLogger(__name__)

# Get shared instances
multi_mcp = get_multi_mcp()


def _resolve_vector_provider(user: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
    """Return (provider, tenant_context) for the calling user.

    Falls back to legacy ``faiss`` MCP path when the tenant has not been
    routed to a cloud-managed vector store.
    """
    tenant_context = resolve_tenant_context(
        request_payload={},
        user=user,
        tenancy_settings=settings.get("tenancy", {}),
    )
    backend_cfg = (settings.get("ingest", {}) or {}).get("vector_store", {}) or {}
    overrides = backend_cfg.get("tenant_overrides") or {}
    provider = (
        overrides.get(tenant_context["tenant_id"])
        or backend_cfg.get("provider")
        or "faiss"
    )
    return str(provider).lower(), tenant_context


def _extract_mcp_text_entries(result: Any) -> list[str]:
    """Extract text fields from MCP CallToolResult-like payloads."""
    entries: list[str] = []
    content = getattr(result, "content", None)
    if not isinstance(content, list):
        return entries
    for item in content:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if isinstance(text, str):
            entries.append(text)
    return entries


def _parse_text_or_list_payload(text: str) -> list[str]:
    """Parse text that may encode a list payload; otherwise return as singleton."""
    raw = (text or "").strip()
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item is not None]
        except Exception:
            continue
    return [text]


# === Document Management Endpoints ===

@router.get("/documents")
async def get_rag_documents():
    """List documents in a recursive tree structure with RAG status"""
    try:
        doc_path = PROJECT_ROOT / "data"
        index_dir = PROJECT_ROOT / "mcp_servers" / "faiss_index"
        
        # Try new ledger format first, fall back to legacy cache
        ledger_file = index_dir / "ledger.json"
        cache_file = index_dir / "doc_index_cache.json"
        
        file_entries = {}  # {path: {hash, status, indexed_at, ...}}
        
        if ledger_file.exists():
            try:
                ledger_data = json.loads(ledger_file.read_text())
                file_entries = ledger_data.get("files", {})
            except:
                pass
        elif cache_file.exists():
            # Legacy format: {"path": "hash"}
            try:
                legacy_cache = json.loads(cache_file.read_text())
                for path, file_hash in legacy_cache.items():
                    file_entries[path] = {
                        "hash": file_hash,
                        "status": "complete",
                        "indexed_at": None,
                        "chunk_count": 0
                    }
            except:
                pass

        def build_tree(path: Path):
            items = []
            # Sort: directories first, then files
            for p in sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if p.name.startswith('.') or p.name in ["__pycache__", "mcp_repos", "faiss_index"]:
                    continue
                
                rel_p = p.relative_to(doc_path).as_posix()
                item = {
                    "name": p.name,
                    "path": rel_p,
                    "type": "folder" if p.is_dir() else p.suffix.lower().replace('.', ''),
                }
                
                if p.is_dir():
                    item["children"] = build_tree(p)
                else:
                    item["size"] = p.stat().st_size
                    entry = file_entries.get(rel_p, {})
                    item["indexed"] = entry.get("status") == "complete" if entry else False
                    item["status"] = entry.get("status", "unindexed")  # New field
                    item["hash"] = entry.get("hash", "Not Indexed")
                    item["chunk_count"] = entry.get("chunk_count", 0)
                    item["error"] = entry.get("error")
                
                items.append(item)
            return items

        files = build_tree(doc_path) if doc_path.exists() else []
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_folder")
async def create_rag_folder(folder_path: str):
    """Create a new folder in RAG documents"""
    try:
        root = PROJECT_ROOT / "data"
        # Sanitize path to allow nested folders but prevent traversal
        clean_path = folder_path.strip("/").replace("..", "")
        target_path = root / clean_path
        
        if target_path.exists():
             raise HTTPException(status_code=400, detail="Folder already exists")
        
        target_path.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "path": str(clean_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
async def delete_rag_item(path: str = Form(...)):
    """Delete a file or folder in RAG documents"""
    try:
        root = PROJECT_ROOT / "data"
        # Sanitize
        clean_path = path.strip("/").replace("..", "")
        target_path = root / clean_path
        
        if not target_path.exists():
            raise HTTPException(status_code=404, detail="Item not found")
            
        # Security check: ensure we are deleting something inside data
        if not str(target_path.resolve()).startswith(str(root.resolve())):
             raise HTTPException(status_code=403, detail="Access denied")

        if target_path.is_dir():
            import shutil
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create_file")
async def create_rag_file(path: str = Form(...), content: str = Form("")):
    """Create a new file in RAG documents"""
    try:
        root = PROJECT_ROOT / "data"
        clean_path = path.strip("/").replace("..", "")
        target_path = root / clean_path
        
        if target_path.exists():
            raise HTTPException(status_code=400, detail="File already exists")
            
        # Ensure parent dir exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        target_path.write_text(content)
        return {"status": "success", "path": str(clean_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rename")
async def rename_rag_item(old_path: str = Form(...), new_path: str = Form(...)):
    """Rename a file or folder in RAG documents"""
    try:
        root = PROJECT_ROOT / "data"
        old_target = root / old_path.strip("/").replace("..", "")
        new_target = root / new_path.strip("/").replace("..", "")
        
        if not old_target.exists():
            raise HTTPException(status_code=404, detail="Item not found")
        if new_target.exists():
            raise HTTPException(status_code=400, detail="Target already exists")
            
        # Security check
        if not str(old_target.resolve()).startswith(str(root.resolve())) or \
           not str(new_target.resolve()).startswith(str(root.resolve())):
             raise HTTPException(status_code=403, detail="Access denied")

        old_target.rename(new_target)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/absolute_path")
async def get_rag_absolute_path(path: str):
    """Get absolute path for a relative RAG path"""
    try:
        root = PROJECT_ROOT / "data"
        target_path = root / path.strip("/").replace("..", "")
        return {"absolute_path": str(target_path.resolve())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/move")
async def move_rag_item(src: str = Form(...), dest: str = Form(...)):
    """Move/Cut-Paste a file or folder"""
    try:
        root = PROJECT_ROOT / "data"
        src_target = root / src.strip("/").replace("..", "")
        dest_target = root / dest.strip("/").replace("..", "")
        
        if not src_target.exists():
            raise HTTPException(status_code=404, detail="Source not found")
            
        # Security check
        if not str(src_target.resolve()).startswith(str(root.resolve())) or \
           not str(dest_target.resolve()).startswith(str(root.resolve())):
             raise HTTPException(status_code=403, detail="Access denied")

        import shutil
        shutil.move(str(src_target), str(dest_target))
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy")
async def copy_rag_item(src: str = Form(...), dest: str = Form(...)):
    """Copy a file or folder"""
    try:
        root = PROJECT_ROOT / "data"
        src_target = root / src.strip("/").replace("..", "")
        dest_target = root / dest.strip("/").replace("..", "")
        
        if not src_target.exists():
            raise HTTPException(status_code=404, detail="Source not found")
            
        # Security check
        if not str(src_target.resolve()).startswith(str(root.resolve())) or \
           not str(dest_target.resolve()).startswith(str(root.resolve())):
             raise HTTPException(status_code=403, detail="Access denied")

        import shutil
        if src_target.is_dir():
            shutil.copytree(src_target, dest_target)
        else:
            shutil.copy2(src_target, dest_target)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_markdown_images(content: str, note_path: Path):
    """
    Scans markdown content for:
    1. Local absolute image paths (/Users/...)
    2. Internet URLs (http://...)
    Moves/resizes them to a local 'attachments' folder relative to the note.
    Returns (updated_content, modified_count)
    """
    pattern = r'!\[(.*?)\]\((.*?)\)'
    modified = False
    new_content = content
    attachments_dir = note_path.parent / "attachments"
    
    matches = re.findall(pattern, content)
    for alt, raw_path in matches:
        img_data = None
        ext = ".png"
        
        # SAFETY: If it's already a localhost API URL, clean it back to relative path
        if "localhost:8000/rag/document_content?path=" in raw_path:
            try:
                from urllib.parse import urlparse, parse_qs
                parsed_url = urlparse(raw_path)
                params = parse_qs(parsed_url.query)
                if 'path' in params:
                    full_img_path = params['path'][0]
                    filename = os.path.basename(full_img_path)
                    rel_path = f"./attachments/{filename}"
                    new_content = new_content.replace(raw_path, rel_path)
                    modified = True
                    continue
            except:
                pass

        is_local = raw_path.startswith("/") or raw_path.startswith("file://")
        is_url = raw_path.startswith("http://") or raw_path.startswith("https://")
        
        if not (is_local or is_url):
            continue
            
        try:
            if is_local:
                file_path_str = raw_path.replace("file://", "")
                img_path = Path(file_path_str)
                if img_path.exists() and img_path.is_file():
                    with open(img_path, "rb") as f:
                        img_data = f.read()
                    ext = img_path.suffix.lower() or ".png"
            elif is_url:
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                response = requests.get(raw_path, timeout=10, headers=headers)
                if response.status_code == 200:
                    img_data = response.content
                    url_ext = Path(raw_path.split('?')[0]).suffix.lower()
                    if url_ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']:
                        ext = url_ext
                    else:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'image/jpeg' in content_type: ext = '.jpg'
                        elif 'image/png' in content_type: ext = '.png'
                        elif 'image/gif' in content_type: ext = '.gif'
                        elif 'image/webp' in content_type: ext = '.webp'
                        elif 'image/svg' in content_type: ext = '.svg'

            if img_data:
                file_hash = hashlib.md5(img_data).hexdigest()
                new_filename = f"img_{file_hash[:10]}{ext}"
                attachments_dir.mkdir(parents=True, exist_ok=True)
                target_file = attachments_dir / new_filename
                
                if not target_file.exists():
                    if ext != '.svg':
                        with Image.open(io.BytesIO(img_data)) as img:
                            max_dim = 1024
                            width, height = img.size
                            if width > max_dim or height > max_dim:
                                if width > height:
                                    new_w, new_h = max_dim, int(height * (max_dim / width))
                                else:
                                    new_h, new_w = max_dim, int(width * (max_dim / height))
                                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                img.save(target_file)
                            else:
                                with open(target_file, "wb") as f:
                                    f.write(img_data)
                    else:
                        with open(target_file, "wb") as f:
                            f.write(img_data)
                
                rel_path = f"./attachments/{new_filename}"
                new_content = new_content.replace(raw_path, rel_path)
                modified = True
        except Exception as e:
            print(f"Error processing image {raw_path}: {e}")
                
    return new_content, modified

@router.post("/save_file")
async def save_rag_file(path: str = Form(...), content: str = Form(...)):
    """Save/Overwrite file content"""
    try:
        root = PROJECT_ROOT / "data"
        clean_path = path.strip("/").replace("..", "")
        target_path = root / clean_path
        
        # Security check
        if not str(target_path.resolve()).startswith(str(root.resolve())):
             raise HTTPException(status_code=403, detail="Access denied")
        
        # Process images if it's a markdown file
        if clean_path.lower().endswith(".md"):
            updated_content, modified = process_markdown_images(content, target_path)
            if modified:
                content = updated_content
        
        target_path.write_text(content)
        return {"status": "success", "content": content if clean_path.lower().endswith(".md") else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_rag_file(
    file: UploadFile = File(...), 
    path: str = Form("")
):
    """Upload a file to RAG documents"""
    try:
        root = PROJECT_ROOT / "data"
        # Sanitize target directory
        target_dir = root
        if path:
            # Prevent directory traversal
            clean_path = path.strip("/").replace("..", "")
            target_dir = root / clean_path
            
        target_dir.mkdir(parents=True, exist_ok=True)
        
        file_location = target_dir / file.filename
        content = await file.read()
        
        with open(file_location, "wb") as f:
            f.write(content)
            
        return {"status": "success", "filename": file.filename, "path": str(file_location.relative_to(root))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Indexing Endpoints ===

@router.post("/reindex")
async def reindex_rag_documents(path: str = None, force: bool = False):
    """Trigger re-indexing of documents via RAG MCP tool"""
    start_ms = now_ms()
    try:
        # Pass the path to the tool if provided
        args = {"target_path": path, "force": force}
        result = await multi_mcp.call_tool("rag", "reindex_documents", args)
        RAG_REQUESTS_TOTAL.labels(endpoint="reindex", status="success").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="reindex").observe(elapsed_ms(start_ms))
        return {"status": "success", "result": result}
    except Exception as e:
        RAG_REQUESTS_TOTAL.labels(endpoint="reindex", status="error").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="reindex").observe(elapsed_ms(start_ms))
        raise HTTPException(status_code=500, detail=f"Failed to trigger reindex: {str(e)}")


@router.get("/indexing_status")
async def get_indexing_status():
    """Get current indexing progress"""
    try:
        result = await multi_mcp.call_tool("rag", "get_indexing_status", {})
        # Parse JSON string from MCP tool
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                if hasattr(item, 'text'):
                    return json.loads(item.text)
        return {
            "active": False,
            "total": 0,
            "completed": 0,
            "currentFile": "",
            "last_error": "",
            "index_manifest": None,
            "index_compatible": None,
            "index_compatibility_reason": "",
        }
    except Exception as e:
        logger.warning("Failed to read indexing status: %s", e)
        return {
            "active": False,
            "total": 0,
            "completed": 0,
            "currentFile": "",
            "last_error": "",
            "index_manifest": None,
            "index_compatible": None,
            "index_compatibility_reason": "",
        }


# === Search Endpoints ===

def find_page_for_chunk(doc_path: str, chunk_text: str) -> int:
    """Lazily find which page contains the chunk text using pymupdf text search."""
    try:
        import pymupdf
        full_path = PROJECT_ROOT / "data" / doc_path
        if not full_path.exists() or not doc_path.endswith('.pdf'):
            return 1  # Default to page 1 for non-PDFs
        
        doc = pymupdf.open(str(full_path))
        # Clean markdown formatting and use first 60 chars for search
        search_text = chunk_text[:150].strip()
        # Remove markdown formatting
        search_text = re.sub(r'\*\*|##|#|\[|\]|\(|\)|!\[|\n', ' ', search_text)
        search_text = re.sub(r'\s+', ' ', search_text).strip()[:60]
        
        if len(search_text) < 10:
            doc.close()
            return 1  # Too short to search

        logger.debug("Page search snippet '%s' in %s", search_text[:40], doc_path)
        for page_num, page in enumerate(doc):
            # Search for text on this page
            if page.search_for(search_text):
                logger.debug("Found match on page %s for %s", page_num + 1, doc_path)
                doc.close()
                return page_num + 1  # 1-indexed
        
        doc.close()
        return 1  # Default to page 1 if not found
    except Exception as e:
        logger.debug("Page lookup failed for %s: %s", doc_path, e)
        return 1


@router.get("/search")
async def rag_search(query: str, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Semantic search against indexed RAG documents with page numbers.

    Delegates to the tenant-configured vector store when ``ingest.vector_store``
    points at a cloud backend (Azure AI Search, AWS OpenSearch, Bedrock KB).
    Falls back to the legacy MCP+FAISS hybrid pipeline for the default tenant
    so the existing local dev path keeps working unchanged.
    """
    start_ms = now_ms()
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    provider, tenant_context = _resolve_vector_provider(user)
    if provider != "faiss":
        try:
            from core.embedding import get_normalized_embedding

            embedding = get_normalized_embedding(query, task_type="search_query")
            store = get_vector_store(tenant_context)
            hits = store.query(
                embedding=embedding.tolist(),
                text=query,
                k=int(settings.get("rag", {}).get("top_k") or 5),
                filters={"tenant_id": tenant_context["tenant_id"]},
            )
            structured_results = [
                {
                    "content": h.text,
                    "source": h.source_uri or h.doc_id,
                    "page": int((h.metadata or {}).get("page") or 1),
                    "score": h.score,
                }
                for h in hits
            ]
            RAG_REQUESTS_TOTAL.labels(endpoint="search", status="success").inc()
            RAG_SEARCH_LATENCY_MS.labels(endpoint="search").observe(elapsed_ms(start_ms))
            RAG_RESULTS_COUNT.labels(endpoint="search").observe(len(structured_results))
            if not structured_results:
                RAG_EMPTY_RESULT_TOTAL.labels(endpoint="search").inc()
            return {
                "status": "success",
                "results": structured_results,
                "vector_provider": provider,
            }
        except Exception as exc:
            RAG_REQUESTS_TOTAL.labels(endpoint="search", status="error").inc()
            RAG_SEARCH_LATENCY_MS.labels(endpoint="search").observe(elapsed_ms(start_ms))
            raise HTTPException(status_code=500, detail=f"Cloud vector search failed: {exc}")
    try:
        args = {"query": query}
        result = await multi_mcp.call_tool("rag", "search_stored_documents_rag", args)
        raw_results: list[str] = []
        for text in _extract_mcp_text_entries(result):
            raw_results.extend(_parse_text_or_list_payload(text))
        
        # Parse results - page navigation handled by frontend search
        structured_results = []
        for r in raw_results:
            # Parse "[Source: path p123]" format
            match = re.search(r'\[Source:\s*(.+?)(?:\s+p(\d+))?\]$', r)
            if match:
                source = match.group(1)
                page_str = match.group(2)
                content = r[:match.start()].strip()
                structured_results.append({
                    "content": content,
                    "source": source,
                    "page": int(page_str) if page_str else 1
                })
            else:
                structured_results.append({
                    "content": r,
                    "source": "unknown",
                    "page": 1
                })
        
        result_count = len(structured_results)
        RAG_REQUESTS_TOTAL.labels(endpoint="search", status="success").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="search").observe(elapsed_ms(start_ms))
        RAG_RESULTS_COUNT.labels(endpoint="search").observe(result_count)
        if result_count == 0:
            RAG_EMPTY_RESULT_TOTAL.labels(endpoint="search").inc()
        return {"status": "success", "results": structured_results}
    except Exception as e:
        logger.exception("RAG search failed: %s", e)
        RAG_REQUESTS_TOTAL.labels(endpoint="search", status="error").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="search").observe(elapsed_ms(start_ms))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/keyword_search")
async def rag_keyword_search(query: str, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Keyword search across document chunks (exact match)"""
    start_ms = now_ms()
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        args = {"query": query}
        result = await multi_mcp.call_tool("rag", "keyword_search", args)
        
        # Extract matches from CallToolResult
        matches: list[str] = []
        for text in _extract_mcp_text_entries(result):
            matches.extend(_parse_text_or_list_payload(text))
        
        match_count = len(matches)
        RAG_REQUESTS_TOTAL.labels(endpoint="keyword_search", status="success").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="keyword_search").observe(elapsed_ms(start_ms))
        RAG_RESULTS_COUNT.labels(endpoint="keyword_search").observe(match_count)
        if match_count == 0:
            RAG_EMPTY_RESULT_TOTAL.labels(endpoint="keyword_search").inc()
        return {"status": "success", "matches": matches}
    except Exception as e:
        RAG_REQUESTS_TOTAL.labels(endpoint="keyword_search", status="error").inc()
        RAG_SEARCH_LATENCY_MS.labels(endpoint="keyword_search").observe(elapsed_ms(start_ms))
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")

@router.get("/ripgrep_search")
async def rag_ripgrep_search(
    query: str,
    regex: bool = False,
    case_sensitive: bool = False,
    target_dir: str = None,
    user: Dict[str, Any] = Depends(require_supabase_user),
):
    """Deep pattern search using ripgrep"""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        args = {"query": query, "regex": regex, "case_sensitive": case_sensitive, "target_dir": target_dir}
        result = await multi_mcp.call_tool("rag", "advanced_ripgrep_search", args)
        
        # Extract results from CallToolResult - ROBUST NOISE-RESISTANT PARSER
        results = []
        
        def extract_json_list(text):
            """Helpful regex to find and extract the JSON list even if there's noise"""
            # Try finding something that looks like a JSON list: [...]
            import re
            match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
            if match:
                 try:
                     return json.loads(match.group(0))
                 except:
                     pass
            
            # Try ast if it looks like a Python list with single quotes
            match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
            if match:
                 try:
                     import ast
                     return ast.literal_eval(match.group(0))
                 except:
                     pass
            return None

        # 1. Try to find content in 'content' list
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                text_content = ""
                if hasattr(item, 'text'):
                    text_content = item.text
                elif isinstance(item, dict) and 'text' in item:
                    text_content = item['text']
                
                if text_content:
                    logger.debug("Ripgrep MCP text payload starts with: %s", text_content[:100])
                    
                    extracted = extract_json_list(text_content)
                    
                    # Validate matches
                    valid_items = []
                    if extracted and isinstance(extracted, list):
                        for x in extracted:
                            if isinstance(x, dict) and "file" in x and "line" in x:
                                valid_items.append(x)
                    
                    if valid_items:
                        results.extend(valid_items)
                    else:
                        # Direct try as fallback
                        try:
                            parsed = json.loads(text_content)
                            if isinstance(parsed, list):
                                valid = [x for x in parsed if isinstance(x, dict) and "file" in x]
                                results.extend(valid)
                            elif isinstance(parsed, dict) and "file" in parsed:
                                results.append(parsed)
                        except:
                            try:
                                import ast
                                parsed = ast.literal_eval(text_content)
                                if isinstance(parsed, list): 
                                    valid = [x for x in parsed if isinstance(x, dict) and "file" in x]
                                    results.extend(valid)
                                elif isinstance(parsed, dict) and "file" in parsed:
                                    results.append(parsed)
                            except:
                                pass
        
        # 2. Fallback: If result itself is already a list (direct return)
        elif isinstance(result, list):
            results = result
            
        logger.debug("Ripgrep router returning %s structured results", len(results))
        if len(results) > 0:
            logger.debug("First ripgrep result sample: %s", results[0])
            
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("Ripgrep router error: %s", e)
        raise HTTPException(status_code=500, detail=f"Ripgrep search failed: {str(e)}")


# === Document Content Endpoints ===

@router.get("/document_chunks")
async def get_document_chunks(path: str):
    """Get cached chunks for a document from the FAISS metadata - FAST, no re-processing."""
    try:
        meta_path = PROJECT_ROOT / "mcp_servers" / "faiss_index" / "metadata.json"
        if not meta_path.exists():
            return {"status": "error", "markdown": "No index found. Please index documents first."}
        
        metadata = json.loads(meta_path.read_text())
        
        # Filter chunks for this document
        doc_chunks = [m["chunk"] for m in metadata if m.get("doc") == path]
        
        if not doc_chunks:
            return {"status": "error", "markdown": f"No chunks found for document: {path}. Try re-indexing."}
        
        # --- BACKEND CAPTION INJECTION ---
        # Load captions.json and replace ![](images/X.png) with actual captions
        try:
            captions_path = meta_path.parent / "captions.json"
            if captions_path.exists():
                captions_ledger = json.loads(captions_path.read_text())
                
                # Define replacer at this scope
                def caption_replacer(match):
                    img_path = match.group(1)  # e.g. "images/file.png"
                    filename = Path(img_path).name  # e.g. "file.png"
                    
                    if filename in captions_ledger and captions_ledger[filename]:
                        caption = captions_ledger[filename]
                        return f"**[Image Caption]:** *{caption}*"
                    return match.group(0)  # Keep original if no caption yet
                
                # Apply regex at outer scope where 're' is accessible
                image_pattern = re.compile(r'!\[.*?\]\((.*?)\)')
                doc_chunks = [image_pattern.sub(caption_replacer, c) for c in doc_chunks]
        except Exception as e:
            logger.warning("Caption injection failed for %s: %s", path, e)
        # ---------------------------------
        
        # Concatenate chunks with separators
        full_text = "\n\n---\n\n".join(doc_chunks)

        # Detect if this is a code file
        code_exts = {
            '.py': 'python', '.tsx': 'typescript', '.ts': 'typescript', 
            '.js': 'javascript', '.jsx': 'javascript', '.html': 'html', 
            '.css': 'css', '.json': 'json', '.c': 'c', '.cpp': 'cpp',
            '.h': 'c', '.hpp': 'cpp', '.md': 'markdown', '.txt': 'text'
        }
        file_ext = Path(path).suffix.lower()
        
        if file_ext in code_exts and file_ext not in ['.md', '.txt']:
            # Wrap in code block
            lang = code_exts[file_ext]
            full_text = f"```{lang}\n{full_text}\n```"
        elif file_ext not in ['.md', '.txt']:
            # Apply heuristics to restore structure from flattened text (DOCS only)
            # 1. Restore headers
            full_text = re.sub(r'\s(#{1,6})\s', r'\n\n\1 ', full_text)
            
            # 2. Add breaks before " **" if it looks like a header
            full_text = re.sub(r'(\.|\:)\s+\*\*', r'\1\n\n**', full_text)

            # 3. Restore Tables: 
            # Pattern A: Header | Separator (Space between)
            full_text = re.sub(r'(\|\s*)(?=\|[:\-]+\|)', r'\1\n', full_text)

            # Pattern B: Separator | Row (Space between)
            full_text = re.sub(r'(\|[:\-]+\|)(\s+)(?=\|)', r'\1\n', full_text)
        
        return {
            "status": "success", 
            "markdown": full_text, 
            "chunks": doc_chunks, 
            "chunk_count": len(doc_chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document_content")
async def get_document_content(path: str):
    """Get the content of a document (binary or text)"""
    try:
        # Check if absolute path first
        if os.path.isabs(path) or path.startswith("/"):
            doc_path = Path(path)
            if not doc_path.exists():
                # Fallback to PROJECT_ROOT / data if it doesn't exist as absolute
                doc_path = PROJECT_ROOT / "data" / path
        else:
            root = PROJECT_ROOT / "data"
            doc_path = root / path
        
        # Fallback: Check relative to PROJECT_ROOT 
        if not doc_path.exists():
            # Try finding it relative to project root
            alt_path = PROJECT_ROOT / path
            if alt_path.exists() and alt_path.is_file():
                doc_path = alt_path
        
        # Deep Search Fallback: If path is just a filename, search for it
        if not doc_path.exists() and len(Path(path).parts) == 1:
            # Search in memory and data
            found_files = list(PROJECT_ROOT.rglob(path))
            # Prioritize memory/ or data/
            for f in found_files:
                if "memory" in str(f) or "data" in str(f):
                    doc_path = f
                    break
            if not doc_path.exists() and found_files:
                doc_path = found_files[0]
            
        if not doc_path.exists():
            raise HTTPException(status_code=404, detail=f"Document not found at {path} or resolved paths")
        
        ext = doc_path.suffix.lower()
        
        # Binary Media
        if ext in ['.pdf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.docx', '.doc']:
            media_types = {
                '.pdf': 'application/pdf',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword'
            }
            return FileResponse(doc_path, media_type=media_types.get(ext, 'application/octet-stream'))
        
        # Simple text extraction for fallback
        content = doc_path.read_text(errors='replace')
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document_preview")
async def get_document_preview(path: str):
    """Get the AI-enhanced markdown version of a document (PDF, DOCX, etc.)"""
    try:
        args = {"path": str(PROJECT_ROOT / "data" / path)}
        # Call the generic preview_document tool
        result = await multi_mcp.call_tool("rag", "preview_document", args)
        
        # 1. Handle stringified JSON results (common in some MCP tool patterns)
        if isinstance(result, str):
            try:
                data = json.loads(result)
                if isinstance(data, dict) and 'markdown' in data:
                    return {"status": "success", "markdown": data['markdown']}
            except:
                pass
            return {"status": "success", "markdown": result}

        # 2. Proper handling of MCP CallToolResult object
        if hasattr(result, 'content') and isinstance(result.content, list):
            for item in result.content:
                text = ""
                if hasattr(item, 'text'):
                    text = item.text
                elif isinstance(item, dict) and 'text' in item:
                    text = item['text']
                
                if text:
                    # Check if the text itself is encoded JSON
                    try:
                        data = json.loads(text)
                        if isinstance(data, dict) and 'markdown' in data:
                            return {"status": "success", "markdown": data['markdown']}
                    except:
                        pass
                    return {"status": "success", "markdown": text}
        
        # 3. Fallback for direct storage objects
        if hasattr(result, 'markdown'):
            return {"status": "success", "markdown": result.markdown}
        
        return {"status": "success", "markdown": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask")
async def ask_rag_document(request: Request, user: Dict[str, Any] = Depends(require_supabase_user)):
    """Interactive chat with a document via RAG with real-time streaming (SSE)"""
    try:
        body = await request.json()
        doc_id = body.get("docId")
        query = body.get("query")
        history = body.get("history", [])
        image = body.get("image") # Base64 image
        debug_reasoning = bool(body.get("debug_reasoning", False))
        
        if not doc_id or not query:
            raise HTTPException(status_code=400, detail="Missing docId or query")
            
        # 1. Get relevant context using MCP tool
        context_results = await multi_mcp.call_tool("rag", "search_stored_documents_rag", {"query": query, "doc_path": doc_id})
        # Extract text from CallToolResult if needed (search_stored_documents_rag returns list)
        context_list = []
        if hasattr(context_results, 'content'):
            for c in context_results.content:
                if hasattr(c, 'text'):
                    try:
                        # The tool returns a list of strings as JSON or raw text
                        import ast
                        parsed = ast.literal_eval(c.text)
                        if isinstance(parsed, list):
                            context_list.extend(parsed)
                        else:
                            context_list.append(c.text)
                    except:
                        context_list.append(c.text)
        
        context_text = "\n\n".join(context_list) if context_list else "No relevant context found in document."

        # 2. Build Ollama Prompt
        tools = body.get("tools")
        project_root = body.get("project_root", "Unknown")
        reasoning_instruction = (
            "CRITICAL: Always start your response with a thinking process enclosed in <think> tags.\n"
            "Analyze the context, identify key sections, and plan your answer before providing the final response."
            if debug_reasoning
            else "Reason internally. Do not reveal chain-of-thought. Return only the final answer with citations."
        )
        system_prompt = f"""You are a helpful document assistant and coding agent. 
Answer the user's question based strictly on the provided context from the document or by using your tools.

CRITICAL: Your current working directory (project root) is: {project_root}
All file operations (read, write, list, etc.) MUST be relative to or within this directory. Do NOT attempt to access files outside of this path.

SHELL ENVIRONMENT:
- You are in a NON-INTERACTIVE shell. 
- NEVER use commands that wait for user input (e.g., `input()` in Python, `read` in bash). 
- If you write scripts, use `sys.argv` to accept arguments.
  Example: `script.py arg1 arg2` instead of interactive prompts.
- Prefer `python3` over `python` for execution.
- If a command hangs, it will be killed after 60 seconds.

If the context doesn't contain the answer, say so, but try to be helpful based on what is available.

{reasoning_instruction}

CONTEXT FROM DOCUMENT:
---
{context_text}
---
"""

        if tools:
            import json
            tools_desc = json.dumps(tools, indent=2)
            system_prompt += f"""
\n### AGENT TOOLS
You have access to the following tools to interact with the environment.
To use a tool, you MUST output a valid JSON block enclosed in markdown code fences, like this:

```json
{{
  "tool": "tool_name",
  "args": {{ "arg_name": "value" }}
}}
```

Available Tools:
{tools_desc}

When the tool output is provided to you in a subsequent message, use it to answer the user's request.
"""
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-5:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        user_msg = {"role": "user", "content": query}
        if image:
            # Strip data:image/png;base64, if present
            if "," in image: image = image.split(",")[1]
            user_msg["images"] = [image]
        messages.append(user_msg)

        model = body.get("model", "qwen3-vl:8b") # Default to Qwen

        async def token_generator():
            try:
                if debug_reasoning:
                    debug_payload = {
                        "debug_reasoning": True,
                        "doc_id": doc_id,
                        "query": query,
                        "retrieved_count": len(context_list),
                        "retrieved_context": context_list,
                    }
                    yield f"event: debug\ndata: {json.dumps(debug_payload)}\n\n"

                # Use a separate session or direct httpx for streaming
                import httpx
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", "http://127.0.0.1:11434/api/chat", json={
                        "model": model, 
                        "messages": messages,
                        "stream": True
                    }) as response:
                        async for line in response.aiter_lines():
                            if not line: continue
                            try:
                                data = json.loads(line)
                                chunk = data.get("message", {}).get("content", "")
                                if chunk:
                                    # SSE format: data: <payload>\n\n
                                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                                if data.get("done"):
                                    break
                            except:
                                continue
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(token_generator(), media_type="text/event-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Image Serving ===

@router.get("/images/{filename}")
async def get_rag_image(filename: str):
    """Serve images extracted by MuPDF/indexing process"""
    try:
        image_path = PROJECT_ROOT / "mcp_servers" / "documents" / "images" / filename
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
