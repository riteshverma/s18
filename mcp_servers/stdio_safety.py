import logging
import sys


def configure_mcp_stdio_logging() -> None:
    """Prevent non-JSON logs from leaking into MCP stdout transport."""
    quiet_loggers = (
        "mcp",
        "mcp.server",
        "mcp.server.lowlevel",
        "mcp.server.fastmcp",
    )

    for name in quiet_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        for handler in logger.handlers:
            if hasattr(handler, "stream"):
                handler.stream = sys.stderr

