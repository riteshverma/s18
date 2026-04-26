__all__ = [
    "DefaultIntegrationAdapter",
    "WiseAIIntegrationAdapter",
    "PowerAppsIntegrationAdapter",
]


def __getattr__(name):
    if name == "DefaultIntegrationAdapter":
        from integrations.adapters.default import DefaultIntegrationAdapter

        return DefaultIntegrationAdapter
    if name == "WiseAIIntegrationAdapter":
        from integrations.adapters.wiseai import WiseAIIntegrationAdapter

        return WiseAIIntegrationAdapter
    if name == "PowerAppsIntegrationAdapter":
        from integrations.adapters.powerapps import PowerAppsIntegrationAdapter

        return PowerAppsIntegrationAdapter
    raise AttributeError(name)
