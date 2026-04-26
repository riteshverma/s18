__all__ = ["DefaultIntegrationAdapter", "WiseAIIntegrationAdapter"]


def __getattr__(name):
    if name == "DefaultIntegrationAdapter":
        from integrations.adapters.default import DefaultIntegrationAdapter

        return DefaultIntegrationAdapter
    if name == "WiseAIIntegrationAdapter":
        from integrations.adapters.wiseai import WiseAIIntegrationAdapter

        return WiseAIIntegrationAdapter
    raise AttributeError(name)
