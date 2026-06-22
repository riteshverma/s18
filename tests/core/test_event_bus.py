import asyncio

from core.event_bus import EventBus


def test_event_bus_tracks_dropped_events():
    async def _run():
        bus = EventBus()
        bus._subscribers = set()
        bus._history.clear()
        bus._drop_counters = {}
        bus.configure(disconnect_on_drop=False, drop_sample_seconds=0.01)
        q = await bus.subscribe(max_queue_size=1)
        await bus.publish("log", "test", {"i": 1})
        await bus.publish("log", "test", {"i": 2})
        return bus.drop_stats().get("log:test", 0), q

    dropped, q = asyncio.run(_run())
    assert dropped >= 1
    assert not q.empty()


def test_event_bus_can_subscribe_without_replaying_history():
    async def _run():
        bus = EventBus()
        bus._subscribers = set()
        bus._history.clear()
        await bus.publish("tool_call", "previous-run", {"phase": "end"})

        q = await bus.subscribe(max_queue_size=10, replay_history=False)
        history_replayed = not q.empty()

        await bus.publish("tool_call", "current-run", {"phase": "start"})
        current_event = await q.get()
        return history_replayed, current_event

    history_replayed, current_event = asyncio.run(_run())
    assert history_replayed is False
    assert current_event["source"] == "current-run"
