import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).parent.parent
LEDGER_DIR = ROOT / "memory" / "usage_ledger"
LEDGER_FILE = LEDGER_DIR / "llm_usage.jsonl"


def append_usage_event(event: Dict[str, Any]) -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.utcnow().isoformat(),
        **(event or {}),
    }
    with open(LEDGER_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
