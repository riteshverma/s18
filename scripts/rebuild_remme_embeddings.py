#!/usr/bin/env python3
"""Re-embed all RemMe memories after changing the embedding model (e.g. Nomic -> BGE).

Usage (from repo root):
  uv run python scripts/rebuild_remme_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remme.store import RemmeStore


def main() -> None:
    store = RemmeStore()
    n = len(store.memories)
    print(f"Re-embedding {n} memories with the current embedding settings...")
    count = store.rebuild_embeddings_from_texts()
    print(f"Done. Indexed {count} vectors -> {store.root / 'index.bin'}")


if __name__ == "__main__":
    main()
