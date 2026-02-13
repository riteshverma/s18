import sqlite3
import json
import uuid
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path

# --- Configuration ---
DB_PATH = Path("data/inbox/notifications.db")
router = APIRouter(prefix="/inbox", tags=["Inbox"])

# --- Schema ---
class Notification(BaseModel):
    id: str
    source: str
    title: str
    body: str  # Markdown support
    priority: int = 1  # 1=Normal, 2=High, 3=Critical
    is_read: bool = False
    timestamp: str
    metadata: Optional[Dict[str, Any]] = {}

class CreateNotificationRequest(BaseModel):
    source: str
    title: str
    body: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = {}

# --- Database Helper ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the SQLite database if it doesn't exist."""
    if not DB_PATH.parent.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            priority INTEGER DEFAULT 1,
            is_read BOOLEAN DEFAULT 0,
            timestamp TEXT NOT NULL,
            metadata TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Initialize on module load
init_db()

# --- Helper Function for Internal Use ---
def send_to_inbox(source: str, title: str, body: str, priority: int = 1, metadata: dict = None) -> str:
    """
    Internal Python API to send a notification to the Inbox.
    Agents and backend services should use this.
    """
    notif_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    meta_json = json.dumps(metadata) if metadata else "{}"
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO notifications (id, source, title, body, priority, is_read, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (notif_id, source, title, body, priority, False, timestamp, meta_json)
    )
    conn.commit()
    conn.close()
    return notif_id

# --- Endpoints ---

INBOX_VIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inbox</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto; padding: 1.5rem; background: #1a1a1a; color: #e0e0e0; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    .sub { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .card { background: #2a2a2a; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; border-left: 4px solid #4a9eff; }
    .card.read { opacity: 0.75; border-left-color: #555; }
    .card.high { border-left-color: #e67e22; }
    .card .source { font-size: 0.8rem; color: #888; margin-bottom: 0.25rem; }
    .card .title { font-weight: 600; margin-bottom: 0.5rem; }
    .card .body { font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; }
    .card .body a { color: #4a9eff; }
    .card .time { font-size: 0.8rem; color: #666; margin-top: 0.5rem; }
    .empty { color: #888; text-align: center; padding: 2rem; }
    .briefing { border-left-color: #2ecc71; }
  </style>
</head>
<body>
  <h1>Inbox</h1>
  <p class="sub">Your morning briefing and other notifications appear here.</p>
  <div id="list">Loading…</div>
  <script>
    async function load() {
      const res = await fetch('/inbox/');
      const items = await res.json();
      const list = document.getElementById('list');
      if (!items.length) { list.innerHTML = '<p class="empty">No notifications yet. Run the briefing: <code>POST /cron/briefing/trigger</code></p>'; return; }
      list.innerHTML = items.map(n => {
        const cls = ['card', n.is_read ? 'read' : '', n.priority >= 2 ? 'high' : '', n.source === 'Morning Briefing' ? 'briefing' : ''].filter(Boolean).join(' ');
        const time = n.timestamp ? new Date(n.timestamp).toLocaleString() : '';
        const body = (n.body || '').replace(/\\n/g, '<br>').replace(/\\[([^\\]]+)\\]\\((https?:[^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
        return `<div class="${cls}"><div class="source">${escapeHtml(n.source)}</div><div class="title">${escapeHtml(n.title)}</div><div class="body">${body}</div><div class="time">${escapeHtml(time)}</div></div>`;
      }).join('');
    }
    function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
    load();
  </script>
</body>
</html>
"""


@router.get("/view", response_class=HTMLResponse)
async def inbox_view():
    """View Inbox in the browser — open this URL to see 'Your morning briefing' and other notifications."""
    return INBOX_VIEW_HTML


@router.get("/", response_model=List[Notification])
async def get_notifications(unread_only: bool = False, limit: int = 50):
    """Fetch notifications, optionally filtering by unread status."""
    conn = get_db_connection()
    c = conn.cursor()
    
    query = "SELECT * FROM notifications"
    params = []
    
    if unread_only:
        query += " WHERE is_read = 0"
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append(Notification(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            body=row["body"],
            priority=row["priority"],
            is_read=bool(row["is_read"]),
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        ))
    return results

@router.post("/", response_model=Dict[str, str])
async def create_notification(request: CreateNotificationRequest):
    """Create a new notification (API Endpoint)."""
    notif_id = send_to_inbox(
        source=request.source,
        title=request.title,
        body=request.body,
        priority=request.priority,
        metadata=request.metadata
    )
    return {"id": notif_id, "status": "created"}

@router.patch("/{notif_id}/read")
async def mark_as_read(notif_id: str):
    """Mark a notification as read."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notif_id,))
    conn.commit()
    found = c.rowcount > 0
    conn.close()
    
    if not found:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "updated"}

@router.delete("/{notif_id}")
async def delete_notification(notif_id: str):
    """Delete a notification."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM notifications WHERE id = ?", (notif_id,))
    conn.commit()
    found = c.rowcount > 0
    conn.close()
    
    if not found:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "deleted"}
