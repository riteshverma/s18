# Shared module -- re-exports from shared.state so `import shared` exposes the API
from .state import (
    PROJECT_ROOT,  # noqa: F401
    active_loops,  # noqa: F401
    get_multi_mcp,  # noqa: F401
    get_remme_store,  # noqa: F401
    get_remme_extractor,  # noqa: F401
)
