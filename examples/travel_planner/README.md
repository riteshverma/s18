# Travel Planner Example

This example shows S18 orchestrating a planning workflow that is unrelated to
healthcare.

## Goal

Generate a 3-day itinerary with budget and logistics checks.

## Run request

```bash
curl -X POST "http://localhost:8000/runs" \
  -H "Content-Type: application/json" \
  -d @examples/travel_planner/run_payload.json
```

## Integration mapping

- `integration_id`: `default`
- `workflow_id`: `travel_itinerary`
- `source_system`: `travel_assistant`

Use this as a starter before creating a dedicated `travel` adapter.
