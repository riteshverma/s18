import asyncio
import importlib
import json
import sys
import threading
import types
from pathlib import Path


class _FakeMCP:
    def __init__(self, *_args, **_kwargs):
        pass

    def tool(self, *args, **_kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

    def run(self):
        pass


class _FakeIndex:
    def __init__(self, vectors=None, dim=3):
        self.vectors = list(vectors or [])
        self.d = dim

    def add(self, vectors):
        self.vectors.extend(list(vectors))

    def copy(self):
        return _FakeIndex(vectors=[list(v) for v in self.vectors], dim=self.d)


def _module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_server_rag_import_stubs(monkeypatch):
    mcp_types = _module("mcp.types", TextContent=object)
    mcp_root = _module("mcp", types=mcp_types)
    mcp_root.__path__ = []

    monkeypatch.setitem(sys.modules, "mcp", mcp_root)
    monkeypatch.setitem(sys.modules, "mcp.types", mcp_types)
    monkeypatch.setitem(sys.modules, "mcp.server", _module("mcp.server"))
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.fastmcp",
        _module("mcp.server.fastmcp", FastMCP=_FakeMCP, Image=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.fastmcp.prompts",
        _module("mcp.server.fastmcp.prompts", base=object()),
    )

    monkeypatch.setitem(sys.modules, "PIL", _module("PIL", Image=object))
    monkeypatch.setitem(sys.modules, "PIL.Image", _module("PIL.Image"))
    monkeypatch.setitem(
        sys.modules,
        "fitz",
        _module(
            "fitz",
            TOOLS=types.SimpleNamespace(
                mupdf_display_errors=lambda *_args, **_kwargs: None,
                set_stderr_log=lambda *_args, **_kwargs: None,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "markitdown",
        _module(
            "markitdown",
            MarkItDown=lambda: types.SimpleNamespace(
                convert=lambda *_args, **_kwargs: types.SimpleNamespace(text_content="")
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", _module("numpy", stack=lambda values: list(values)))
    monkeypatch.setitem(sys.modules, "pymupdf4llm", _module("pymupdf4llm"))
    monkeypatch.setitem(sys.modules, "trafilatura", _module("trafilatura"))
    monkeypatch.setitem(sys.modules, "tqdm", _module("tqdm", tqdm=lambda iterable, **_kwargs: iterable))
    monkeypatch.setitem(sys.modules, "rank_bm25", _module("rank_bm25", BM25Okapi=object))

    settings = {"rag": {"chunk_size": 1000, "chunk_overlap": 100, "max_chunk_length": 2000, "top_k": 5}}
    monkeypatch.setitem(
        sys.modules,
        "config.settings_loader",
        _module(
            "config.settings_loader",
            settings=settings,
            get_embedding_provider=lambda: "ollama",
            get_llama_cpp_timeout=lambda: 30,
            get_llama_cpp_url=lambda *_args, **_kwargs: "http://localhost",
            get_model=lambda *_args, **_kwargs: "test-model",
            get_ollama_url=lambda *_args, **_kwargs: "http://localhost",
            get_timeout=lambda: 30,
            load_settings=lambda: {"models": {"embedding_provider": "ollama"}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.faiss_runtime",
        _module(
            "core.faiss_runtime",
            create_index_flat_l2=lambda dim: _FakeIndex(dim=dim),
            get_faiss=lambda: object(),
            read_index=lambda _path: None,
            write_index=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.embedding",
        _module(
            "core.embedding",
            get_normalized_embedding=lambda *_args, **_kwargs: [0.0, 0.0, 0.0],
            try_get_normalized_embedding=lambda *_args, **_kwargs: [0.0, 0.0, 0.0],
            get_batch_normalized_embeddings=lambda texts, **_kwargs: [[0.0, 0.0, 0.0] for _ in texts],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.model_manager",
        _module("core.model_manager", ModelManager=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.rag_rerank",
        _module(
            "core.rag_rerank",
            RerankCandidate=object,
            build_reranker=lambda *_args, **_kwargs: None,
            load_rerank_config=lambda *_args, **_kwargs: {},
        ),
    )

    model_names = (
        "AddInput",
        "AddOutput",
        "SqrtInput",
        "SqrtOutput",
        "StringsToIntsInput",
        "StringsToIntsOutput",
        "ExpSumInput",
        "ExpSumOutput",
        "PythonCodeInput",
        "PythonCodeOutput",
        "UrlInput",
        "FilePathInput",
        "MarkdownInput",
        "MarkdownOutput",
        "ChunkListOutput",
        "SearchDocumentsInput",
    )
    monkeypatch.setitem(sys.modules, "models", _module("models", **{name: object for name in model_names}))


def _import_server_rag(monkeypatch):
    _install_server_rag_import_stubs(monkeypatch)
    sys.modules.pop("mcp_servers.server_rag", None)
    return importlib.import_module("mcp_servers.server_rag")


def test_index_images_reloads_index_artifacts_after_captioning(tmp_path: Path, monkeypatch):
    server_rag = _import_server_rag(monkeypatch)

    fake_server_file = tmp_path / "server_rag.py"
    fake_server_file.write_text("# test server module", encoding="utf-8")
    images_dir = tmp_path / "documents" / "images"
    index_dir = tmp_path / "faiss_index"
    images_dir.mkdir(parents=True)
    index_dir.mkdir()
    (images_dir / "doc-page-1.png").write_bytes(b"image")
    (index_dir / "captions.json").write_text("{}", encoding="utf-8")
    metadata_file = index_dir / "metadata.json"
    index_file = index_dir / "index.bin"
    metadata_file.write_text("[]", encoding="utf-8")

    state = {
        "current_index": _FakeIndex(vectors=[]),
        "written_index": None,
        "manifest_saved": False,
    }

    def caption_image(_path):
        metadata_file.write_text(
            json.dumps([{"doc": "fresh-doc.md", "chunk": "fresh scheduler chunk"}]),
            encoding="utf-8",
        )
        index_file.write_bytes(b"fresh-index")
        state["current_index"] = _FakeIndex(vectors=[[9.0, 9.0, 9.0]])
        return "fresh caption"

    monkeypatch.setattr(server_rag, "__file__", str(fake_server_file))
    monkeypatch.setattr(server_rag, "caption_image", caption_image)
    monkeypatch.setattr(server_rag, "get_embedding", lambda _text: [1.0, 2.0, 3.0])
    monkeypatch.setattr(server_rag, "RAG_INDEX_LOCK", threading.RLock())
    monkeypatch.setattr(server_rag, "np", types.SimpleNamespace(stack=lambda values: list(values)))
    monkeypatch.setattr(server_rag, "create_index_flat_l2", lambda dim: _FakeIndex(dim=dim))
    monkeypatch.setattr(server_rag, "faiss_read_index", lambda _path: state["current_index"].copy())

    def write_index(index, _path):
        state["written_index"] = index.copy()

    monkeypatch.setattr(server_rag, "faiss_write_index", write_index)
    monkeypatch.setattr(
        server_rag,
        "_save_index_manifest",
        lambda *_args, **_kwargs: state.__setitem__("manifest_saved", True),
    )

    result = asyncio.run(server_rag.index_images())

    assert result == "Successfully processed 1 images. Index updated."
    docs = [entry["doc"] for entry in json.loads(metadata_file.read_text(encoding="utf-8"))]
    assert docs == ["fresh-doc.md", "doc-page-1.png"]
    assert state["written_index"].vectors == [[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]]
    assert state["manifest_saved"] is True
