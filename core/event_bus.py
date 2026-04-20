import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from collections import deque
import weakref
import time

logger = logging.getLogger("event_bus")


class EventBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = set()
            cls._instance._history = deque(maxlen=100)
            cls._instance._drop_counters = {}
            cls._instance._last_drop_log_at = 0.0
            cls._instance._drop_sample_seconds = 1.0
            cls._instance._disconnect_on_drop = False
        return cls._instance

    def configure(self, *, disconnect_on_drop: bool = False, drop_sample_seconds: float = 1.0):
        self._disconnect_on_drop = disconnect_on_drop
        self._drop_sample_seconds = max(0.1, float(drop_sample_seconds))

    def drop_stats(self) -> Dict[str, int]:
        return dict(self._drop_counters)

    async def publish(self, event_type: str, source: str, data: Dict[str, Any]):
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "source": source,
            "data": data,
        }

        self._history.append(event)

        dead = []

        for ref in list(self._subscribers):
            q = ref()
            if q is None:
                dead.append(ref)
                continue

            try:
                q.put_nowait(event)  # Non-blocking
            except asyncio.QueueFull:
                key = f"{event_type}:{source}"
                self._drop_counters[key] = self._drop_counters.get(key, 0) + 1
                now = time.monotonic()
                if now - self._last_drop_log_at >= self._drop_sample_seconds:
                    logger.warning(
                        "Dropping event due to full subscriber queue type=%s source=%s dropped=%s subscribers=%s",
                        event_type,
                        source,
                        self._drop_counters[key],
                        len(self._subscribers),
                    )
                    self._last_drop_log_at = now
                if self._disconnect_on_drop:
                    dead.append(ref)

        # Cleanup dead references
        for ref in dead:
            self._subscribers.discard(ref)

    async def subscribe(self, max_queue_size: int = 100):
        q = asyncio.Queue(maxsize=max_queue_size)
        self._subscribers.add(weakref.ref(q))

        # Replay last 5 events
        for event in list(self._history)[-5:]:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                break

        return q

    def unsubscribe(self, q: asyncio.Queue):
        for ref in list(self._subscribers):
            if ref() is q:
                self._subscribers.discard(ref)


# Singleton instance used by all consumers
event_bus = EventBus()
