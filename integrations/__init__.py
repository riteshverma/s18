from integrations.contracts import CanonicalRunRequest

__all__ = ["CanonicalRunRequest", "get_integration_adapter"]


def get_integration_adapter(*args, **kwargs):
    from integrations.registry import get_integration_adapter as _get_integration_adapter

    return _get_integration_adapter(*args, **kwargs)
