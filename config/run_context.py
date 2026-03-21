"""Per-async-task context for run metadata (e.g. source_system for Wise vs S18 model routing)."""

from contextvars import ContextVar

source_system_var: ContextVar[str] = ContextVar("source_system", default="s18")


def set_run_source_system(value: str):
    """Return a token for reset(); value is normalized to lowercase."""
    normalized = (value or "s18").strip().lower()
    return source_system_var.set(normalized)


def reset_run_source_system(token) -> None:
    source_system_var.reset(token)


def get_run_source_system() -> str:
    return source_system_var.get()
