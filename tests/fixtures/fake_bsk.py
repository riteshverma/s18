"""Minimal bsk CLI stub used by tests/test_browserskill_server.py.

Echoes the argv it received as JSON so the proxy's argv construction,
flag rendering and env plumbing can be asserted without the real Rust
binary. ``fail`` exits non-zero and ``hang`` sleeps, to exercise the
error and timeout paths.
"""

import json
import os
import sys
import time


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "fail":
        sys.stderr.write("stub failure\n")
        return 3
    if args and args[0] == "hang":
        time.sleep(30)
        return 0
    print(
        json.dumps(
            {
                "argv": args,
                "bsk_home": os.environ.get("BSK_HOME"),
                "bsk_auto_start": os.environ.get("BSK_AUTO_START"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
