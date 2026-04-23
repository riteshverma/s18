# s18_engine

`s18_engine` is the productized core import surface for orchestration and
canonical integration contracts.

## Current scope

- `AgentLoop4` execution engine
- Canonical run contract models
- Integration adapter resolution API

## Example

```python
from s18_engine import AgentLoop4, CanonicalRunRequest, get_integration_adapter

adapter = get_integration_adapter("default")
request = CanonicalRunRequest(query="Plan a 3-day trip")
```
