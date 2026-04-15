import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).parent.parent
    config_path = root / "mcp_servers" / "mcp_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    gbrain = config.get("gbrain")
    if not isinstance(gbrain, dict):
        print("FAIL: missing 'gbrain' server entry in mcp_config.json")
        return 1

    expected = {
        "command": "gbrain",
        "args": ["serve"],
        "enabled": False,
    }
    for key, value in expected.items():
        if gbrain.get(key) != value:
            print(f"FAIL: gbrain.{key} expected {value!r}, got {gbrain.get(key)!r}")
            return 1

    print("PASS: gbrain MCP registration template is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
