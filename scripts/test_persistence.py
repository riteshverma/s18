import sys
import json
from types import SimpleNamespace

sys.path.append(".")
from core.persistence import persistence_manager, SNAPSHOT_FILE
# Mock active_loops
from shared import state

def test_persistence():
    print("Testing Persistence Manager...")
    
    # 1. Mock State
    print("📝 Mocking active runs...")
    mock_loop = SimpleNamespace(
        context=SimpleNamespace(
            plan_graph=SimpleNamespace(
                graph={
                    "status": "running", 
                    "original_query": "Find the answer",
                    "session_id": "mock_run_123"
                }
            )
        )
    )
    state.active_loops["mock_run_123"] = mock_loop
    
    # 2. Save Snapshot
    print("💾 Saving snapshot...")
    persistence_manager.save_snapshot()
    
    if SNAPSHOT_FILE.exists():
        print("✅ Snapshot file created.")
    else:
        print("❌ Snapshot file missing!")
        sys.exit(1)
        
    # 3. Verify Content
    data = json.loads(SNAPSHOT_FILE.read_text())
    runs = data.get("active_runs", [])
    print(f"📄 Found {len(runs)} runs in snapshot.")
    
    if len(runs) > 0 and runs[0]["run_id"] == "mock_run_123":
        print("✅ Run data verified.")
        print(f"   Query: {runs[0]['query']}")
    else:
        print("❌ Run data mismatch.")
        
    # 4. Clean up mock
    del state.active_loops["mock_run_123"]
    SNAPSHOT_FILE.unlink(missing_ok=True)
    
    print("🎉 Persistence Test Complete.")

if __name__ == "__main__":
    from core.logging_setup import configure_logging
    configure_logging()
    test_persistence()
