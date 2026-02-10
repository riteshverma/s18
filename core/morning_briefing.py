"""
Morning Briefing: scheduled job that fetches overnight news, summarizes,
sends to Inbox, and publishes to EventBus. Combines Scheduler + Inbox + EventBus.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Any

logger = logging.getLogger("morning_briefing")

# Overnight = items from the last N hours (e.g. since previous evening)
BRIEFING_HOURS = 12
MAX_ITEMS_IN_SUMMARY = 25


def _parse_ts(ts: str) -> datetime:
    """Parse ISO timestamp; return now on failure."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def _is_after_cutoff(dt: datetime, cutoff: datetime) -> bool:
    """Compare timestamps; normalize to naive local for comparison."""
    if dt.tzinfo:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt >= cutoff


async def _fetch_overnight_items() -> List[Any]:
    """Fetch news from configured sources and return items from the last BRIEFING_HOURS."""
    from routers.news import get_news_settings, fetch_hn, fetch_rss

    news_settings = get_news_settings()
    sources = news_settings["sources"]
    cutoff = datetime.now() - timedelta(hours=BRIEFING_HOURS)
    all_items = []

    for source in sources:
        if not source.get("enabled", True):
            continue
        try:
            if source["id"] == "hn":
                items = await fetch_hn()
            elif source["type"] == "rss" and source.get("feed_url"):
                items = await fetch_rss(source)
            else:
                continue
            for item in items:
                try:
                    dt = _parse_ts(item.timestamp)
                    if _is_after_cutoff(dt, cutoff):
                        all_items.append(item)
                except Exception:
                    all_items.append(item)
        except Exception as e:
            logger.warning("Briefing: skip source %s: %s", source.get("name", source["id"]), e)

    all_items.sort(key=lambda x: x.timestamp, reverse=True)
    return all_items[:MAX_ITEMS_IN_SUMMARY]


def _build_summary_markdown(items: List[Any]) -> str:
    """Build a markdown body for the inbox notification."""
    if not items:
        return "No overnight items in the last {} hours.".format(BRIEFING_HOURS)

    lines = [
        f"**Overnight summary** (last {BRIEFING_HOURS} hours, {len(items)} items):",
        "",
    ]
    by_source: dict = {}
    for item in items:
        name = item.source_name
        if name not in by_source:
            by_source[name] = []
        by_source[name].append(item)

    for source_name, source_items in by_source.items():
        lines.append(f"### {source_name}")
        for item in source_items:
            title = (item.title or "No title")[:100]
            url = getattr(item, "url", "") or ""
            lines.append(f"- [{title}]({url})")
        lines.append("")

    lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    return "\n".join(lines)


async def run_morning_briefing() -> dict:
    """
    Fetch overnight news, build summary, send to Inbox, and publish to EventBus.
    Returns dict with notification_id, item_count, and success/error.
    """
    from core.event_bus import event_bus
    from routers.inbox import send_to_inbox

    run_id = f"briefing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Starting morning briefing: %s", run_id)

    await event_bus.publish(
        "log",
        "morning_briefing",
        {"message": "Morning briefing started.", "metadata": {"run_id": run_id}},
    )

    try:
        items = await _fetch_overnight_items()
        body = _build_summary_markdown(items)
        notif_id = send_to_inbox(
            source="Morning Briefing",
            title="Your morning briefing",
            body=body,
            priority=1,
            metadata={"run_id": run_id, "item_count": len(items)},
        )

        await event_bus.publish(
            "briefing_ready",
            "morning_briefing",
            {
                "message": "Morning briefing delivered to Inbox.",
                "notification_id": notif_id,
                "item_count": len(items),
                "run_id": run_id,
            },
        )
        await event_bus.publish(
            "success",
            "morning_briefing",
            {"message": "Briefing completed.", "metadata": {"job_id": "morning_briefing", "run_id": run_id}},
        )

        logger.info("Morning briefing done: %s items -> inbox %s", len(items), notif_id)
        return {"notification_id": notif_id, "item_count": len(items), "run_id": run_id, "error": None}
    except Exception as e:
        logger.exception("Morning briefing failed: %s", e)
        await event_bus.publish(
            "error",
            "morning_briefing",
            {"message": str(e), "run_id": run_id},
        )
        send_to_inbox(
            source="Morning Briefing",
            title="Morning briefing failed",
            body=f"Error: {str(e)}",
            priority=2,
            metadata={"run_id": run_id},
        )
        return {"notification_id": None, "item_count": 0, "run_id": run_id, "error": str(e)}
