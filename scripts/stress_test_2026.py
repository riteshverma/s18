import asyncio
import time
import uuid
import random
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup
sys.path.append(".")
from core.scheduler import scheduler_service
from routers.inbox import send_to_inbox, init_db, get_db_connection
from core.event_bus import event_bus
from core.persistence import persistence_manager

# Configure Logging
logging.basicConfig(level=logging.WARNING) # Silence internal logs
logger = logging.getLogger("STRESSTEST")
logger.setLevel(logging.INFO)

# --- MOCKS ---
class MockAgentRunner:
    async def run_agent(self, agent_type, input_data):
        # Fake "Thinking" Latency
        await asyncio.sleep(random.uniform(0.01, 0.05)) 
        return {
            "success": True,
            "output": {"thought": "I am a mock agent", "result": "Stress Test Passed"}
        }

# --- TESTS ---

async def stress_inbox(count: int = 1000):
    logger.info(f"📨 inbox: Stressing with {count} messages...")
    start = time.time()
    
    # 1. Burst Write
    ids = []
    for i in range(count):
        notif_id = send_to_inbox(
            source=f"Stress_{i}", 
            title=f"Msg {i}", 
            body="Load Test", 
            priority=1
        )
        ids.append(notif_id)
        
    duration = time.time() - start
    rate = count / duration
    logger.info(f"✅ inbox: Wrote {count} in {duration:.2f}s ({rate:.0f} msg/s)")
    
    # 2. Burst DB Read
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT count(*) FROM notifications")
    total = c.fetchone()[0]
    conn.close()
    
    if total >= count:
        logger.info(f"✅ inbox: Verification passed (DB count: {total})")
    else:
        logger.error(f"❌ inbox: Data loss detected! Expected >={count}, got {total}")

    # Cleanup (Clean the specific test data)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM notifications WHERE source LIKE 'Stress_%'")
    conn.commit()
    conn.close()

async def stress_event_bus(count: int = 5000):
    logger.info(f"📡 event_bus: Stressing with {count} events...")
    
    # 1. Subscribe
    queue = await event_bus.subscribe()
    
    # 2. Async Publish Task
    start = time.time()
    
    producer_done = asyncio.Event()

    async def producer():
        for i in range(count):
            await event_bus.publish("stress_test", "tester", {"seq": i})
            # No sleep, max throughput
        producer_done.set()
            
    # 3. Consumer Task
    received = 0
    async def consumer():
        nonlocal received
        # Drain until producer is done and queue remains empty for a short period.
        while True:
            if producer_done.is_set() and queue.empty():
                break
            try:
                await asyncio.wait_for(queue.get(), timeout=0.5)
                received += 1
            except asyncio.TimeoutError:
                if producer_done.is_set() and queue.empty():
                    break
            
    await asyncio.gather(producer(), consumer())
    
    duration = time.time() - start
    processed_rate = received / duration if duration > 0 else 0
    dropped = max(count - received, 0)
    logger.info(
        f"✅ event_bus: Delivered {received}/{count} in {duration:.2f}s "
        f"({processed_rate:.0f} events/s), dropped≈{dropped}"
    )
    
    event_bus.unsubscribe(queue)

async def stress_scheduler(count: int = 50):
    logger.info(f"⏰ scheduler: Scheduling {count} concurrent jobs...")
    
    scheduler_service.initialize()
    
    created_jobs = []
    
    # 1. Create many jobs with same schedule
    for i in range(count):
        # Schedule for "every minute" so they are active but not necessarily triggering immediately in this short test
        # We just want to test the scheduler's memory/overhead
        job = scheduler_service.add_job(
            name=f"StressJob_{i}",
            cron_expression="* * * * *",
            agent_type="MockAgent",
            query="stress"
        )
        created_jobs.append(job.id)
        
    # 2. Verify List Performance
    start = time.time()
    jobs = scheduler_service.list_jobs()
    duration = time.time() - start
    
    if len(jobs) >= count:
        logger.info(f"✅ scheduler: Listed {len(jobs)} jobs in {duration:.4f}s")
    else:
        logger.error(f"❌ scheduler: Job count mismatch")
        
    # Cleanup
    for jid in created_jobs:
        scheduler_service.delete_job(jid)
    logger.info("✅ scheduler: Cleanup complete")

async def stress_persistence():
    logger.info("💾 persistence: Validating Snapshot logic...")
    
    # Simulate heavy state
    from shared import state
    from types import SimpleNamespace
    
    # Create 100 fake running loops
    track_ids = []
    for i in range(100):
        rid = f"run_{uuid.uuid4()}"
        state.active_loops[rid] = SimpleNamespace(
            context=SimpleNamespace(
                plan_graph=SimpleNamespace(
                    graph={"status": "running", "original_query": "stress test persistence"}
                )
            )
        )
        track_ids.append(rid)
        
    # Snapshot
    start = time.time()
    persistence_manager.save_snapshot()
    duration = time.time() - start
    logger.info(f"✅ persistence: Saved 100 loops in {duration:.4f}s")
    
    # Cleanup State
    for rid in track_ids:
        del state.active_loops[rid]
        
    # Load Snapshot
    persistence_manager.load_snapshot()
    logger.info("✅ persistence: Load verified (No crash)")

async def main():
    print("🚀 STARTING 2026 STRESS TEST SUITE")
    print("===================================")
    
    try:
        init_db()
        
        await stress_inbox(1000)
        print("-----------------------------------")
        
        await stress_event_bus(10000)
        print("-----------------------------------")
        
        await stress_scheduler(100)
        print("-----------------------------------")
        
        await stress_persistence()
        print("-----------------------------------")
        
        print("✨ ALL SYSTEMS GO. 2026 READY.")
        
    except Exception as e:
        logger.error(f"❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
