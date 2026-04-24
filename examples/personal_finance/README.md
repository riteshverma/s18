# Personal Finance Agent Example

This example demonstrates how to run S18 with a non-medical workflow.

## Goal

Use the same S18 orchestration stack to:
- summarize monthly spending
- detect unusual transactions
- suggest budgeting actions

## Run request

```bash
curl -X POST "http://localhost:8000/runs" \
  -H "Content-Type: application/json" \
  -d @examples/personal_finance/run_payload.json
```

## Integration mapping

- `integration_id`: `default`
- `workflow_id`: `finance_budgeting`
- `source_system`: `finance_app`

This intentionally uses the default adapter to show that the same core can be
reused before you create a dedicated adapter.
