import json
from types import SimpleNamespace

from core import persistence as persistence_module


def test_persistence_snapshot_includes_active_runs(tmp_path):
    snapshot_file = tmp_path / "snapshot.json"
    original_snapshot_file = persistence_module.SNAPSHOT_FILE
    original_active_loops = dict(persistence_module.active_loops)
    try:
        persistence_module.SNAPSHOT_FILE = snapshot_file
        persistence_module.active_loops.clear()
        fake_graph = SimpleNamespace(graph={"status": "running", "original_query": "demo"})
        persistence_module.active_loops["run-1"] = SimpleNamespace(context=SimpleNamespace(plan_graph=fake_graph))
        persistence_module.PersistenceManager.save_snapshot()
        payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
        assert payload["active_runs"][0]["run_id"] == "run-1"
        assert payload["active_runs"][0]["query"] == "demo"
    finally:
        persistence_module.SNAPSHOT_FILE = original_snapshot_file
        persistence_module.active_loops.clear()
        persistence_module.active_loops.update(original_active_loops)
