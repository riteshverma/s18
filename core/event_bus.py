import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
from collections import deque
import weakref

logger = logging.getLogger("event_bus")


class EventBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = set()
            cls._instance._history = deque(maxlen=100)
        return cls._instance

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
                logger.warning("Dropping event due to full subscriber queue")

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
