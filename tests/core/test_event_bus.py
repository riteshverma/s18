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
